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

// Orchestrates the full text-to-image pipeline:
//   text encoding → denoising loop → VAE decode
class Pipeline {
  public:
    Pipeline() = default;

    // Load all engines and tokenizer
    bool init(const PipelineConfig& config);

    // Generate an image from a text prompt
    // Returns RGB pixel data (512*512*3 bytes)
    std::vector<uint8_t> generate(const std::string& prompt, uint32_t seed = 42,
                                   int num_steps = 4);

  private:
    // Pipeline phases
    std::vector<float> encode_text(const std::string& prompt);
    std::vector<float> denoise(const std::vector<float>& text_embeddings,
                                uint32_t seed, int num_steps);
    std::vector<uint8_t> decode_latents(const std::vector<float>& latents);

    // Init random latents from seed
    std::vector<float> init_latents(uint32_t seed);

    ClipTokenizer tokenizer_;
    LcmScheduler scheduler_;

    TrtEngine text_encoder_;
    TrtEngine unet_;
    TrtEngine vae_decoder_;

    // Whether to load/unload engines sequentially to save VRAM
    bool sequential_mode_ = false;

    PipelineConfig config_;
};
