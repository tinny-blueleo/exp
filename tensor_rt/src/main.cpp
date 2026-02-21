#include "pipeline.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --prompt TEXT   Text prompt (default: \"Taj Mahal in space\")\n"
              << "  --seed N        Random seed (default: 42)\n"
              << "  --steps N       Number of LCM steps (default: 4)\n"
              << "  -o FILE         Output PNG path (default: output.png)\n"
              << "  --engine-dir D  Engine directory (default: engines/)\n"
              << "  --vocab FILE    BPE vocab file (default: data/bpe_simple_vocab_16e6.txt)\n"
              << "  -h, --help      Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string prompt = "Taj Mahal in space";
    uint32_t seed = 42;
    int steps = 4;
    std::string output = "output.png";
    std::string engine_dir = "engines";
    std::string vocab = "data/bpe_simple_vocab_16e6.txt";

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (std::strcmp(argv[i], "--engine-dir") == 0 && i + 1 < argc) {
            engine_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab = argv[++i];
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    PipelineConfig config;
    config.text_encoder_engine = engine_dir + "/text_encoder.trt";
    config.unet_engine = engine_dir + "/unet.trt";
    config.vae_decoder_engine = engine_dir + "/vae_decoder.trt";
    config.vocab_path = vocab;

    Pipeline pipeline;
    if (!pipeline.init(config)) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return 1;
    }

    auto pixels = pipeline.generate(prompt, seed, steps);

    if (pixels.empty()) {
        std::cerr << "Generation failed" << std::endl;
        return 1;
    }

    // Write PNG
    int ok = stbi_write_png(output.c_str(), 512, 512, 3, pixels.data(),
                            512 * 3);
    if (ok) {
        std::cout << "Saved: " << output << std::endl;
    } else {
        std::cerr << "Failed to write: " << output << std::endl;
        return 1;
    }

    return 0;
}
