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

// FP16 ↔ FP32 conversion helpers
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
            uint32_t result = sign;
            float f;
            std::memcpy(&f, &result, sizeof(f));
            return f;
        }
        exp = 1;
        while (!(frac & 0x400)) {
            frac <<= 1;
            exp--;
        }
        frac &= 0x3FF;
    } else if (exp == 31) {
        uint32_t result = sign | 0x7F800000 | (frac << 13);
        float f;
        std::memcpy(&f, &result, sizeof(f));
        return f;
    }

    uint32_t result = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
}

// Compute guidance scale embedding for LCM (sinusoidal positional encoding).
// Matches diffusers get_guidance_scale_embedding / SimianLuo get_w_embedding.
static std::vector<float> get_guidance_scale_embedding(float w,
                                                        int embedding_dim = 256) {
    w = w * 1000.0f;
    int half_dim = embedding_dim / 2;
    float log_base = std::log(10000.0f) / (float)(half_dim - 1);

    std::vector<float> emb(embedding_dim);
    for (int i = 0; i < half_dim; i++) {
        float freq = std::exp((float)i * -log_base);
        emb[i] = std::sin(w * freq);
        emb[half_dim + i] = std::cos(w * freq);
    }
    return emb;
}

// Helper: find an input by name, return nullptr if not found
static const BufferInfo* find_input(const TrtEngine& engine,
                                     const std::string& name) {
    for (const auto& inp : engine.inputs()) {
        if (inp.name == name) return &inp;
    }
    return nullptr;
}

// Helper: find an output by name, return nullptr if not found
static const BufferInfo* find_output(const TrtEngine& engine,
                                      const std::string& name) {
    for (const auto& out : engine.outputs()) {
        if (out.name == name) return &out;
    }
    return nullptr;
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;

    if (!tokenizer_.load(config.vocab_path)) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return false;
    }

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

std::vector<float> Pipeline::encode_text(const std::string& prompt) {
    double t0 = now_ms();

    if (sequential_mode_ && !text_encoder_.is_loaded()) {
        text_encoder_.load(config_.text_encoder_engine);
    }

    auto token_ids = tokenizer_.encode(prompt);
    std::cout << "Tokens: ";
    for (int i = 0; i < std::min((int)token_ids.size(), 10); i++)
        std::cout << token_ids[i] << " ";
    std::cout << "... (" << token_ids.size() << " total)" << std::endl;

    // Set input shape for dynamic engines
    nvinfer1::Dims ids_shape{2, {1, 77}};
    text_encoder_.set_input_shape("input_ids", ids_shape);

    text_encoder_.set_input("input_ids", token_ids.data(),
                            token_ids.size() * sizeof(int32_t));
    text_encoder_.infer();

    // Get output embeddings [1, 77, 768]
    const int embed_size = 1 * 77 * 768;

    const BufferInfo* out_info = find_output(text_encoder_, "last_hidden_state");
    if (!out_info && !text_encoder_.outputs().empty()) {
        out_info = &text_encoder_.outputs()[0];
    }

    std::vector<float> embeddings(embed_size);

    if (out_info->dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> fp16_data(embed_size);
        text_encoder_.get_output(out_info->name, fp16_data.data(),
                                 embed_size * sizeof(uint16_t));
        for (int i = 0; i < embed_size; i++) {
            embeddings[i] = fp16_to_fp32(fp16_data[i]);
        }
    } else {
        text_encoder_.get_output(out_info->name, embeddings.data(),
                                 embed_size * sizeof(float));
    }

    if (sequential_mode_) {
        text_encoder_.unload();
    }

    double elapsed = now_ms() - t0;
    std::cout << "Text encoding: " << elapsed << " ms" << std::endl;
    return embeddings;
}

std::vector<float> Pipeline::init_latents(uint32_t seed) {
    const int latent_size = 1 * 4 * 64 * 64;
    std::vector<float> latents(latent_size);

    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < latent_size; i++) {
        latents[i] = dist(gen);
    }

    return latents;
}

std::vector<float> Pipeline::denoise(const std::vector<float>& text_embeddings,
                                      uint32_t seed, int num_steps) {
    double t0 = now_ms();

    if (sequential_mode_ && !unet_.is_loaded()) {
        unet_.load(config_.unet_engine);
    }

    scheduler_.set_timesteps(num_steps);

    auto latents = init_latents(seed);
    const int latent_size = (int)latents.size();

    // Compute guidance scale embedding (constant across all steps)
    const float guidance_scale = 8.0f;
    auto w_embedding = get_guidance_scale_embedding(guidance_scale, 256);

    bool has_timestep_cond = (find_input(unet_, "timestep_cond") != nullptr);

    // Determine if UNet expects FP16 inputs
    bool unet_fp16 = false;
    const BufferInfo* sample_info = find_input(unet_, "sample");
    if (sample_info && sample_info->dtype == nvinfer1::DataType::kHALF) {
        unet_fp16 = true;
    }

    // Set dynamic input shapes
    nvinfer1::Dims sample_shape{4, {1, 4, 64, 64}};
    nvinfer1::Dims timestep_shape{1, {1}};
    nvinfer1::Dims embed_shape{3, {1, 77, 768}};
    unet_.set_input_shape("sample", sample_shape);
    unet_.set_input_shape("timestep", timestep_shape);
    unet_.set_input_shape("encoder_hidden_states", embed_shape);
    if (has_timestep_cond) {
        nvinfer1::Dims cond_shape{2, {1, 256}};
        unet_.set_input_shape("timestep_cond", cond_shape);
    }

    // Set guidance scale embedding (constant across steps)
    if (has_timestep_cond) {
        const BufferInfo* cond_info = find_input(unet_, "timestep_cond");
        if (cond_info && cond_info->dtype == nvinfer1::DataType::kHALF) {
            std::vector<uint16_t> w_fp16(256);
            for (int i = 0; i < 256; i++)
                w_fp16[i] = fp32_to_fp16(w_embedding[i]);
            unet_.set_input("timestep_cond", w_fp16.data(),
                           256 * sizeof(uint16_t));
        } else {
            unet_.set_input("timestep_cond", w_embedding.data(),
                           256 * sizeof(float));
        }
    }

    // Set text embeddings (constant across steps)
    int embed_size = (int)text_embeddings.size();
    if (unet_fp16) {
        std::vector<uint16_t> embed_fp16(embed_size);
        for (int j = 0; j < embed_size; j++)
            embed_fp16[j] = fp32_to_fp16(text_embeddings[j]);
        unet_.set_input("encoder_hidden_states", embed_fp16.data(),
                       embed_size * sizeof(uint16_t));
    } else {
        unet_.set_input("encoder_hidden_states", text_embeddings.data(),
                       text_embeddings.size() * sizeof(float));
    }

    // Find output tensor name
    const BufferInfo* unet_out = find_output(unet_, "out_sample");
    if (!unet_out && !unet_.outputs().empty()) {
        unet_out = &unet_.outputs()[0];
    }

    for (int i = 0; i < scheduler_.num_steps(); i++) {
        double step_t0 = now_ms();
        int t = scheduler_.timestep(i);

        if (unet_fp16) {
            std::vector<uint16_t> latents_fp16(latent_size);
            for (int j = 0; j < latent_size; j++)
                latents_fp16[j] = fp32_to_fp16(latents[j]);
            unet_.set_input("sample", latents_fp16.data(),
                           latent_size * sizeof(uint16_t));
        } else {
            unet_.set_input("sample", latents.data(),
                           latent_size * sizeof(float));
        }

        int64_t timestep_val = t;
        unet_.set_input("timestep", &timestep_val, sizeof(int64_t));

        unet_.infer();

        std::vector<float> noise_pred(latent_size);

        if (unet_out->dtype == nvinfer1::DataType::kHALF) {
            std::vector<uint16_t> fp16_out(latent_size);
            unet_.get_output(unet_out->name, fp16_out.data(),
                            latent_size * sizeof(uint16_t));
            for (int j = 0; j < latent_size; j++)
                noise_pred[j] = fp16_to_fp32(fp16_out[j]);
        } else {
            unet_.get_output(unet_out->name, noise_pred.data(),
                            latent_size * sizeof(float));
        }

        scheduler_.step(noise_pred.data(), i, latents.data(), latent_size, seed);

        double step_elapsed = now_ms() - step_t0;
        std::cout << "  Step " << i + 1 << "/" << num_steps << " (t=" << t
                  << "): " << step_elapsed << " ms" << std::endl;
    }

    if (sequential_mode_) {
        unet_.unload();
    }

    double elapsed = now_ms() - t0;
    std::cout << "Denoising: " << elapsed << " ms" << std::endl;
    return latents;
}

std::vector<uint8_t> Pipeline::decode_latents(
    const std::vector<float>& latents) {
    double t0 = now_ms();

    if (sequential_mode_ && !vae_decoder_.is_loaded()) {
        vae_decoder_.load(config_.vae_decoder_engine);
    }

    const int latent_size = (int)latents.size();

    // Scale latents by 1/vae_scaling_factor before decoding.
    // The HF ONNX VAE expects pre-scaled input.
    const float vae_scaling_factor = 0.18215f;
    std::vector<float> scaled_latents(latent_size);
    for (int i = 0; i < latent_size; i++) {
        scaled_latents[i] = latents[i] / vae_scaling_factor;
    }

    // Set dynamic input shape
    nvinfer1::Dims latent_shape{4, {1, 4, 64, 64}};

    const BufferInfo* inp_info = find_input(vae_decoder_, "latent_sample");
    if (!inp_info && !vae_decoder_.inputs().empty()) {
        inp_info = &vae_decoder_.inputs()[0];
    }

    vae_decoder_.set_input_shape(inp_info->name, latent_shape);

    if (inp_info->dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> fp16_latents(latent_size);
        for (int i = 0; i < latent_size; i++)
            fp16_latents[i] = fp32_to_fp16(scaled_latents[i]);
        vae_decoder_.set_input(inp_info->name, fp16_latents.data(),
                               latent_size * sizeof(uint16_t));
    } else {
        vae_decoder_.set_input(inp_info->name, scaled_latents.data(),
                               latent_size * sizeof(float));
    }

    vae_decoder_.infer();

    // Get decoded image [1, 3, 512, 512]
    const int img_elements = 1 * 3 * 512 * 512;

    const BufferInfo* out_info = find_output(vae_decoder_, "sample");
    if (!out_info && !vae_decoder_.outputs().empty()) {
        out_info = &vae_decoder_.outputs()[0];
    }

    std::vector<float> raw_image(img_elements);
    if (out_info->dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> fp16_out(img_elements);
        vae_decoder_.get_output(out_info->name, fp16_out.data(),
                                img_elements * sizeof(uint16_t));
        for (int i = 0; i < img_elements; i++)
            raw_image[i] = fp16_to_fp32(fp16_out[i]);
    } else {
        vae_decoder_.get_output(out_info->name, raw_image.data(),
                                img_elements * sizeof(float));
    }

    if (sequential_mode_) {
        vae_decoder_.unload();
    }

    // Convert from CHW float [-1, 1] to HWC uint8 [0, 255]
    const int H = 512, W = 512, C = 3;
    std::vector<uint8_t> pixels(H * W * C);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                float val = raw_image[c * H * W + y * W + x];
                val = (val + 1.0f) * 0.5f * 255.0f;
                val = std::clamp(val, 0.0f, 255.0f);
                pixels[(y * W + x) * C + c] = (uint8_t)(val + 0.5f);
            }
        }
    }

    double elapsed = now_ms() - t0;
    std::cout << "VAE decode: " << elapsed << " ms" << std::endl;
    return pixels;
}

std::vector<uint8_t> Pipeline::generate(const std::string& prompt,
                                         uint32_t seed, int num_steps) {
    double t0 = now_ms();
    std::cout << "\nGenerating: \"" << prompt << "\" (seed=" << seed
              << ", steps=" << num_steps << ")" << std::endl;

    auto embeddings = encode_text(prompt);
    auto latents = denoise(embeddings, seed, num_steps);
    auto pixels = decode_latents(latents);

    double total = now_ms() - t0;
    std::cout << "\nTotal generation: " << total << " ms" << std::endl;
    return pixels;
}
