#pragma once

#include "engine.h"
#include "scheduler.h"
#include "tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

struct PipelineConfig {
    std::string text_encoder_engine;
    std::string unet_engine;
    std::string vae_decoder_engine;
    std::string vocab_path;
};

// Stable Diffusion text-to-image pipeline using TensorRT.
//
// How diffusion works (simplified):
//
//   You can't directly generate a 512x512 image from text — that's a
//   786,432-dimensional space with no structure. Instead, diffusion models
//   work in a small "latent space" (64x64x4 = 16,384 values) and learn to
//   reverse the process of adding noise.
//
//   The pipeline has three phases:
//
//   1. TEXT ENCODING (CLIP): Convert the text prompt into a sequence of
//      77 embedding vectors (each 768-dim) that the UNet can understand.
//      This is like translating English into the model's internal language.
//
//   2. DENOISING (UNet): Start with pure random noise in latent space.
//      Over several steps, ask the UNet "what noise do you see?" and
//      subtract it. Each step, the UNet is told the current timestep
//      (how noisy the image still is) and the text embeddings (what to
//      generate). After 4 LCM steps, the noise becomes a clean latent
//      image encoding.
//
//   3. DECODING (VAE): The Variational Autoencoder expands the tiny
//      64x64x4 latent representation back into a full 512x512x3 RGB
//      image. This is a learned upsampler — not a simple resize.
//
//   LCM (Latent Consistency Model) is a distilled version of Stable
//   Diffusion that needs only 4 steps instead of the usual 20-50,
//   because it was trained to take bigger jumps during denoising.
class Pipeline {
  public:
    Pipeline() = default;

    bool init(const PipelineConfig& config);

    // Returns RGB pixel data (512*512*3 bytes).
    std::vector<uint8_t> generate(const std::string& prompt,
                                   uint32_t seed = 42, int num_steps = 4);

  private:
    std::vector<float> encode_text(const std::string& prompt);
    std::vector<float> denoise(const std::vector<float>& text_embeddings,
                                uint32_t seed, int num_steps);
    std::vector<uint8_t> decode_latents(const std::vector<float>& latents);
    std::vector<float> init_latents(uint32_t seed);

    ClipTokenizer tokenizer_;
    LcmScheduler scheduler_;

    TrtEngine text_encoder_;
    TrtEngine unet_;
    TrtEngine vae_decoder_;

    // On GPUs with limited VRAM (e.g. 4GB), all three engines may not fit
    // simultaneously. Sequential mode loads/unloads one engine at a time.
    bool sequential_mode_ = false;
    PipelineConfig config_;
};
