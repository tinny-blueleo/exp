#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch())
        .count();
}

// ── FP16 ↔ FP32 conversion ────────────────────────────────────────────────
// TensorRT engines built with --fp16 use half-precision for internal compute
// and may expose FP16 I/O tensors. We work in FP32 on the CPU side and
// convert at the boundary.

static uint16_t fp32_to_fp16(float value) {
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (f >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | frac;
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) {
            uint32_t r = sign; float f; std::memcpy(&f, &r, 4); return f;
        }
        exp = 1;
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
    } else if (exp == 31) {
        uint32_t r = sign | 0x7F800000 | (frac << 13);
        float f; std::memcpy(&f, &r, 4); return f;
    }
    uint32_t r = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float f; std::memcpy(&f, &r, 4); return f;
}

// ── Guidance scale embedding ───────────────────────────────────────────────
// LCM distills classifier-free guidance (CFG) into the model itself. Instead
// of running the UNet twice per step (conditioned + unconditioned) and
// blending, LCM takes a guidance scale value (e.g. 8.0) and encodes it as a
// 256-dim sinusoidal vector — the same kind of positional encoding used for
// diffusion timesteps. This vector tells the UNet how strongly to follow the
// text prompt. It's computed once and reused for every denoising step.

static std::vector<float> get_guidance_scale_embedding(float w,
                                                        int embedding_dim = 256) {
    w *= 1000.0f;
    int half = embedding_dim / 2;
    float log_base = std::log(10000.0f) / (float)(half - 1);

    std::vector<float> emb(embedding_dim);
    for (int i = 0; i < half; i++) {
        float freq = std::exp((float)i * -log_base);
        emb[i] = std::sin(w * freq);
        emb[half + i] = std::cos(w * freq);
    }
    return emb;
}

// ── Tensor lookup helpers ──────────────────────────────────────────────────

static const BufferInfo* find_input(const TrtEngine& engine,
                                     const std::string& name) {
    for (const auto& inp : engine.inputs())
        if (inp.name == name) return &inp;
    return nullptr;
}

static const BufferInfo* find_output(const TrtEngine& engine,
                                      const std::string& name) {
    for (const auto& out : engine.outputs())
        if (out.name == name) return &out;
    return nullptr;
}

// ── Pipeline init ──────────────────────────────────────────────────────────

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;

    if (!tokenizer_.load(config.vocab_path)) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return false;
    }

    // Try loading all three engines into VRAM at once. If the VAE doesn't
    // fit (common on 4GB GPUs since text_enc=470MB + UNet=1.6GB already
    // uses ~2.1GB), fall back to loading one engine at a time.
    if (!text_encoder_.load(config.text_encoder_engine)) return false;
    if (!unet_.load(config.unet_engine)) return false;
    if (!vae_decoder_.load(config.vae_decoder_engine)) {
        std::cout << "VRAM limited — switching to sequential engine loading"
                  << std::endl;
        sequential_mode_ = true;
        unet_.unload();
        text_encoder_.unload();
    }

    scheduler_.init();

    std::cout << "Pipeline initialized"
              << (sequential_mode_ ? " (sequential mode)" : "") << std::endl;
    return true;
}

// ── Phase 1: Text Encoding ─────────────────────────────────────────────────
// The CLIP text encoder converts a text prompt into a dense representation
// that the UNet understands. The process:
//
//   "Taj Mahal in space"
//       ↓ tokenizer (BPE)
//   [49406, 24805, 22301, 530, 2138, 49407, 49407, ...]  (77 ints)
//       ↓ text encoder (transformer)
//   [1, 77, 768] float tensor — one 768-dim vector per token position
//
// The encoder produces a "last_hidden_state" — the contextual embedding of
// each token, where each token's vector is influenced by all other tokens
// via self-attention. This is what guides the UNet during denoising.

std::vector<float> Pipeline::encode_text(const std::string& prompt) {
    double t0 = now_ms();

    if (sequential_mode_ && !text_encoder_.is_loaded())
        text_encoder_.load(config_.text_encoder_engine);

    // Tokenize: convert text → integer token IDs, padded to 77 with EOT.
    auto token_ids = tokenizer_.encode(prompt);
    std::cout << "Tokens: ";
    for (int i = 0; i < std::min((int)token_ids.size(), 10); i++)
        std::cout << token_ids[i] << " ";
    std::cout << "... (" << token_ids.size() << " total)" << std::endl;

    // Set shape and run the text encoder.
    nvinfer1::Dims ids_shape{2, {1, 77}};
    text_encoder_.set_input_shape("input_ids", ids_shape);
    text_encoder_.set_input("input_ids", token_ids.data(),
                            token_ids.size() * sizeof(int32_t));
    text_encoder_.infer();

    // Read back the contextual embeddings [1, 77, 768].
    const int embed_size = 1 * 77 * 768;
    const BufferInfo* out = find_output(text_encoder_, "last_hidden_state");
    if (!out) out = &text_encoder_.outputs()[0];

    std::vector<float> embeddings(embed_size);
    if (out->dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> fp16(embed_size);
        text_encoder_.get_output(out->name, fp16.data(), embed_size * 2);
        for (int i = 0; i < embed_size; i++)
            embeddings[i] = fp16_to_fp32(fp16[i]);
    } else {
        text_encoder_.get_output(out->name, embeddings.data(), embed_size * 4);
    }

    if (sequential_mode_) text_encoder_.unload();

    std::cout << "Text encoding: " << now_ms() - t0 << " ms" << std::endl;
    return embeddings;
}

// ── Phase 2: Denoising (UNet) ──────────────────────────────────────────────
// This is the core of diffusion. We start with pure Gaussian noise and
// iteratively denoise it, guided by the text embeddings.
//
// At each step:
//   1. The UNet receives the current noisy latents, the timestep (an integer
//      saying "how noisy is this?"), the text embeddings, and the guidance
//      scale embedding.
//   2. It predicts the noise component in the latents.
//   3. The scheduler uses this prediction to compute a cleaner version of
//      the latents, removing some of the noise.
//
// After 4 steps, the latents represent a clean image encoding that the VAE
// can decode into pixels.
//
// LCM is special because normal Stable Diffusion needs 20-50 steps.
// LCM was trained via "consistency distillation" to jump directly from noisy
// to clean in fewer steps, using a learned guidance embedding instead of the
// expensive two-pass CFG trick.

std::vector<float> Pipeline::init_latents(uint32_t seed) {
    // Latent space: [1, 4, 64, 64] — 4 channels at 1/8th image resolution.
    // This is the compressed representation the UNet operates on.
    const int latent_size = 1 * 4 * 64 * 64;
    std::vector<float> latents(latent_size);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < latent_size; i++)
        latents[i] = dist(gen);
    return latents;
}

std::vector<float> Pipeline::denoise(const std::vector<float>& text_embeddings,
                                      uint32_t seed, int num_steps) {
    double t0 = now_ms();

    if (sequential_mode_ && !unet_.is_loaded())
        unet_.load(config_.unet_engine);

    scheduler_.set_timesteps(num_steps);
    auto latents = init_latents(seed);
    const int latent_size = (int)latents.size();

    // Guidance scale = 8.0 tells the model "follow the text prompt strongly."
    // Values closer to 1.0 = more creative/random, higher = more literal.
    auto w_embedding = get_guidance_scale_embedding(8.0f, 256);

    // Detect I/O properties from the engine.
    bool has_timestep_cond = (find_input(unet_, "timestep_cond") != nullptr);
    const BufferInfo* sample_buf = find_input(unet_, "sample");
    bool unet_fp16 = sample_buf && sample_buf->dtype == nvinfer1::DataType::kHALF;

    // Tell TensorRT the concrete shapes for this inference.
    unet_.set_input_shape("sample", {4, {1, 4, 64, 64}});
    unet_.set_input_shape("timestep", {1, {1}});
    unet_.set_input_shape("encoder_hidden_states", {3, {1, 77, 768}});
    if (has_timestep_cond)
        unet_.set_input_shape("timestep_cond", {2, {1, 256}});

    // Upload inputs that stay constant across all denoising steps.
    if (has_timestep_cond) {
        const BufferInfo* cond = find_input(unet_, "timestep_cond");
        if (cond->dtype == nvinfer1::DataType::kHALF) {
            std::vector<uint16_t> fp16(256);
            for (int i = 0; i < 256; i++) fp16[i] = fp32_to_fp16(w_embedding[i]);
            unet_.set_input("timestep_cond", fp16.data(), 256 * 2);
        } else {
            unet_.set_input("timestep_cond", w_embedding.data(), 256 * 4);
        }
    }

    int embed_size = (int)text_embeddings.size();
    if (unet_fp16) {
        std::vector<uint16_t> fp16(embed_size);
        for (int j = 0; j < embed_size; j++)
            fp16[j] = fp32_to_fp16(text_embeddings[j]);
        unet_.set_input("encoder_hidden_states", fp16.data(), embed_size * 2);
    } else {
        unet_.set_input("encoder_hidden_states", text_embeddings.data(),
                        embed_size * 4);
    }

    const BufferInfo* unet_out = find_output(unet_, "out_sample");
    if (!unet_out) unet_out = &unet_.outputs()[0];

    // ── Denoising loop ──
    // Each iteration: upload noisy latents → UNet predicts noise → scheduler
    // subtracts noise and optionally adds fresh noise for the next step.
    for (int i = 0; i < scheduler_.num_steps(); i++) {
        double step_t0 = now_ms();
        int t = scheduler_.timestep(i);

        // Upload current latents.
        if (unet_fp16) {
            std::vector<uint16_t> fp16(latent_size);
            for (int j = 0; j < latent_size; j++)
                fp16[j] = fp32_to_fp16(latents[j]);
            unet_.set_input("sample", fp16.data(), latent_size * 2);
        } else {
            unet_.set_input("sample", latents.data(), latent_size * 4);
        }

        // Upload the timestep — tells the UNet how much noise to expect.
        // High t (999) = very noisy, low t (259) = mostly clean.
        int64_t timestep_val = t;
        unet_.set_input("timestep", &timestep_val, sizeof(int64_t));

        unet_.infer();

        // Read the predicted noise.
        std::vector<float> noise_pred(latent_size);
        if (unet_out->dtype == nvinfer1::DataType::kHALF) {
            std::vector<uint16_t> fp16(latent_size);
            unet_.get_output(unet_out->name, fp16.data(), latent_size * 2);
            for (int j = 0; j < latent_size; j++)
                noise_pred[j] = fp16_to_fp32(fp16[j]);
        } else {
            unet_.get_output(unet_out->name, noise_pred.data(), latent_size * 4);
        }

        // Scheduler step: subtract predicted noise, apply LCM boundary
        // conditions, and inject fresh noise for non-final steps.
        scheduler_.step(noise_pred.data(), i, latents.data(), latent_size, seed);

        std::cout << "  Step " << i + 1 << "/" << num_steps
                  << " (t=" << t << "): " << now_ms() - step_t0
                  << " ms" << std::endl;
    }

    if (sequential_mode_) unet_.unload();

    std::cout << "Denoising: " << now_ms() - t0 << " ms" << std::endl;
    return latents;
}

// ── Phase 3: VAE Decode ────────────────────────────────────────────────────
// The VAE (Variational Autoencoder) decoder expands the 64x64x4 latent
// representation into a 512x512x3 RGB image. It's essentially a learned
// neural upsampler that understands the structure of natural images.
//
// The latent space was trained so that similar images have nearby latent
// codes. This is why diffusion works in latent space — the UNet only needs
// to navigate a 16K-dimensional space instead of a 786K-dimensional pixel
// space, making it much more tractable.
//
// Scaling factor: The VAE encoder compressed values by multiplying by 0.18215
// during training. We must undo this (divide by 0.18215) before decoding.
// The HF ONNX model expects this pre-scaling; our old custom export baked
// it into the ONNX graph, but the official HF model does not.

std::vector<uint8_t> Pipeline::decode_latents(
    const std::vector<float>& latents) {
    double t0 = now_ms();

    if (sequential_mode_ && !vae_decoder_.is_loaded())
        vae_decoder_.load(config_.vae_decoder_engine);

    const int latent_size = (int)latents.size();

    // Undo the VAE encoder's scaling factor.
    const float vae_scaling_factor = 0.18215f;
    std::vector<float> scaled(latent_size);
    for (int i = 0; i < latent_size; i++)
        scaled[i] = latents[i] / vae_scaling_factor;

    const BufferInfo* inp = find_input(vae_decoder_, "latent_sample");
    if (!inp) inp = &vae_decoder_.inputs()[0];

    vae_decoder_.set_input_shape(inp->name, {4, {1, 4, 64, 64}});

    if (inp->dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> fp16(latent_size);
        for (int i = 0; i < latent_size; i++)
            fp16[i] = fp32_to_fp16(scaled[i]);
        vae_decoder_.set_input(inp->name, fp16.data(), latent_size * 2);
    } else {
        vae_decoder_.set_input(inp->name, scaled.data(), latent_size * 4);
    }

    vae_decoder_.infer();

    // Read the decoded image: [1, 3, 512, 512] in CHW format, range [-1, 1].
    const int img_elements = 1 * 3 * 512 * 512;
    const BufferInfo* out = find_output(vae_decoder_, "sample");
    if (!out) out = &vae_decoder_.outputs()[0];

    std::vector<float> raw(img_elements);
    if (out->dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> fp16(img_elements);
        vae_decoder_.get_output(out->name, fp16.data(), img_elements * 2);
        for (int i = 0; i < img_elements; i++)
            raw[i] = fp16_to_fp32(fp16[i]);
    } else {
        vae_decoder_.get_output(out->name, raw.data(), img_elements * 4);
    }

    if (sequential_mode_) vae_decoder_.unload();

    // Convert from neural network format (CHW float, range [-1,1]) to image
    // format (HWC uint8, range [0,255]).
    // CHW = channels-first: all red values, then all green, then all blue.
    // HWC = interleaved:    RGBRGBRGB... (what PNG/display expects).
    const int H = 512, W = 512, C = 3;
    std::vector<uint8_t> pixels(H * W * C);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                float val = raw[c * H * W + y * W + x];  // CHW → pixel
                val = (val + 1.0f) * 0.5f * 255.0f;      // [-1,1] → [0,255]
                val = std::clamp(val, 0.0f, 255.0f);
                pixels[(y * W + x) * C + c] = (uint8_t)(val + 0.5f);
            }
        }
    }

    std::cout << "VAE decode: " << now_ms() - t0 << " ms" << std::endl;
    return pixels;
}

// ── Top-level generate ─────────────────────────────────────────────────────

std::vector<uint8_t> Pipeline::generate(const std::string& prompt,
                                         uint32_t seed, int num_steps) {
    double t0 = now_ms();
    std::cout << "\nGenerating: \"" << prompt << "\" (seed=" << seed
              << ", steps=" << num_steps << ")" << std::endl;

    auto embeddings = encode_text(prompt);
    auto latents = denoise(embeddings, seed, num_steps);
    auto pixels = decode_latents(latents);

    std::cout << "\nTotal generation: " << now_ms() - t0 << " ms" << std::endl;
    return pixels;
}
