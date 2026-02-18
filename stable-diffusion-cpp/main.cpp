#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "stable-diffusion.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

static void log_cb(sd_log_level_t level, const char* text, void* data) {
    const char* level_str = "?????";
    switch (level) {
        case SD_LOG_DEBUG: level_str = "DEBUG"; break;
        case SD_LOG_INFO:  level_str = "INFO";  break;
        case SD_LOG_WARN:  level_str = "WARN";  break;
        case SD_LOG_ERROR: level_str = "ERROR"; break;
    }
    fprintf(stderr, "[%-5s] %s", level_str, text);
}

static void progress_cb(int step, int steps, float time, void* data) {
    fprintf(stderr, "  step %d/%d (%.1fs)\n", step, steps, time);
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s -m <model> [options]\n"
        "\n"
        "Required:\n"
        "  -m, --model <path>           Path to model (.gguf, .safetensors, .ckpt)\n"
        "\n"
        "Optional:\n"
        "  --diffusion-model <path>     Path to standalone diffusion model\n"
        "  --vae <path>                 Path to standalone VAE model\n"
        "  --clip_l <path>              Path to clip-l text encoder\n"
        "  --clip_g <path>              Path to clip-g text encoder\n"
        "  --t5xxl <path>               Path to T5-XXL text encoder\n"
        "  -p, --prompt <text>          Prompt (default: \"a lovely cat\")\n"
        "  -n, --negative-prompt <text> Negative prompt (default: \"\")\n"
        "  -W, --width <int>            Image width (default: 512)\n"
        "  -H, --height <int>           Image height (default: 512)\n"
        "  --steps <int>                Sampling steps (default: 20)\n"
        "  --cfg-scale <float>          CFG scale (default: 7.0)\n"
        "  --seed <int>                 RNG seed (default: 42, -1 for random)\n"
        "  --threads <int>              Thread count (default: auto)\n"
        "  -o, --output <path>          Output file (default: output.png)\n"
        "  -h, --help                   Show this help\n",
        prog);
}

int main(int argc, char* argv[]) {
    // Defaults
    std::string model_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string prompt          = "a lovely cat";
    std::string negative_prompt;
    std::string output_path     = "output.png";
    int width       = 512;
    int height      = 512;
    int steps       = 20;
    float cfg_scale = 7.0f;
    int64_t seed    = 42;
    int n_threads   = -1;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model")                model_path           = next();
        else if (arg == "--diffusion-model")                diffusion_model_path = next();
        else if (arg == "--vae")                            vae_path             = next();
        else if (arg == "--clip_l")                         clip_l_path          = next();
        else if (arg == "--clip_g")                         clip_g_path          = next();
        else if (arg == "--t5xxl")                          t5xxl_path           = next();
        else if (arg == "-p" || arg == "--prompt")          prompt               = next();
        else if (arg == "-n" || arg == "--negative-prompt") negative_prompt      = next();
        else if (arg == "-W" || arg == "--width")           width                = std::stoi(next());
        else if (arg == "-H" || arg == "--height")          height               = std::stoi(next());
        else if (arg == "--steps")                          steps                = std::stoi(next());
        else if (arg == "--cfg-scale")                      cfg_scale            = std::stof(next());
        else if (arg == "--seed")                           seed                 = std::stoll(next());
        else if (arg == "--threads" || arg == "-t")         n_threads            = std::stoi(next());
        else if (arg == "-o" || arg == "--output")          output_path          = next();
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() && diffusion_model_path.empty()) {
        fprintf(stderr, "error: --model or --diffusion-model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Set up logging and progress
    sd_set_log_callback(log_cb, nullptr);
    sd_set_progress_callback(progress_cb, nullptr);

    if (n_threads <= 0) {
        n_threads = sd_get_num_physical_cores();
    }
    fprintf(stderr, "threads: %d\n", n_threads);
    fprintf(stderr, "system:  %s\n", sd_get_system_info());

    // Initialize context
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path           = model_path.c_str();
    ctx_params.diffusion_model_path = diffusion_model_path.c_str();
    ctx_params.vae_path             = vae_path.c_str();
    ctx_params.clip_l_path          = clip_l_path.c_str();
    ctx_params.clip_g_path          = clip_g_path.c_str();
    ctx_params.t5xxl_path           = t5xxl_path.c_str();
    ctx_params.n_threads            = n_threads;
    ctx_params.vae_decode_only      = true;
    ctx_params.free_params_immediately = true;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create sd context\n");
        return 1;
    }

    // Set up generation params
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt          = prompt.c_str();
    gen_params.negative_prompt = negative_prompt.c_str();
    gen_params.width           = width;
    gen_params.height          = height;
    gen_params.seed            = seed;
    gen_params.batch_count     = 1;
    gen_params.sample_params.sample_steps       = steps;
    gen_params.sample_params.guidance.txt_cfg   = cfg_scale;

    // Use model defaults for sampler/scheduler
    gen_params.sample_params.sample_method = sd_get_default_sample_method(ctx);
    gen_params.sample_params.scheduler     = sd_get_default_scheduler(ctx, gen_params.sample_params.sample_method);

    fprintf(stderr, "prompt:    \"%s\"\n", prompt.c_str());
    fprintf(stderr, "size:      %dx%d\n", width, height);
    fprintf(stderr, "steps:     %d\n", steps);
    fprintf(stderr, "cfg_scale: %.1f\n", cfg_scale);
    fprintf(stderr, "seed:      %lld\n", (long long)seed);
    fprintf(stderr, "sampler:   %s\n", sd_sample_method_name(gen_params.sample_params.sample_method));
    fprintf(stderr, "scheduler: %s\n", sd_scheduler_name(gen_params.sample_params.scheduler));

    // Generate
    sd_image_t* results = generate_image(ctx, &gen_params);
    if (!results) {
        fprintf(stderr, "error: image generation failed\n");
        free_sd_ctx(ctx);
        return 1;
    }

    // Save output
    int ok = stbi_write_png(output_path.c_str(),
                            results[0].width, results[0].height,
                            results[0].channel, results[0].data, 0);
    if (ok) {
        fprintf(stderr, "saved: %s (%ux%u)\n", output_path.c_str(), results[0].width, results[0].height);
    } else {
        fprintf(stderr, "error: failed to write %s\n", output_path.c_str());
    }

    // Cleanup
    free(results[0].data);
    free(results);
    free_sd_ctx(ctx);

    return ok ? 0 : 1;
}
