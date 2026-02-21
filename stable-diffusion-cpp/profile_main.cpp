/**
 * Profile stable-diffusion.cpp inference.
 *
 * Loads model once, does a warmup generation, then profiles a second
 * generation with per-phase timing (from library logs) and memory stats.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "stable-diffusion.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include <cuda_runtime.h>

struct GpuMem {
    size_t free_bytes;
    size_t total_bytes;
    double used_mb() const { return (total_bytes - free_bytes) / (1024.0 * 1024.0); }
    double free_mb() const { return free_bytes / (1024.0 * 1024.0); }
    double total_mb() const { return total_bytes / (1024.0 * 1024.0); }
};

static GpuMem get_gpu_mem() {
    GpuMem m{};
    cudaMemGetInfo(&m.free_bytes, &m.total_bytes);
    return m;
}

static double get_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            // "VmRSS:    123456 kB"
            size_t kb = 0;
            sscanf(line.c_str(), "VmRSS: %zu", &kb);
            return kb / 1024.0;
        }
    }
    return 0;
}

static bool g_profiling = false;

static void log_cb(sd_log_level_t level, const char* text, void* data) {
    if (!g_profiling && level < SD_LOG_WARN) return;
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
    if (g_profiling)
        fprintf(stderr, "  step %d/%d (%.1fs)\n", step, steps, time);
}

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main() {
    const char* model_path = "models/LCM_Dreamshaper_v7-f16.gguf";
    const char* prompt      = "Taj Mahal in space";
    const int   steps       = 4;
    const float cfg_scale   = 1.0f;
    const int64_t seed      = 42;

    sd_set_log_callback(log_cb, nullptr);
    sd_set_progress_callback(progress_cb, nullptr);

    int n_threads = sd_get_num_physical_cores();

    // --- Model loading ---
    fprintf(stderr, "\n========== MODEL LOADING ==========\n");
    auto t_load_start = Clock::now();

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path           = model_path;
    ctx_params.diffusion_model_path = "";
    ctx_params.vae_path             = "";
    ctx_params.clip_l_path          = "";
    ctx_params.clip_g_path          = "";
    ctx_params.t5xxl_path           = "";
    ctx_params.n_threads            = n_threads;
    ctx_params.vae_decode_only      = true;
    ctx_params.free_params_immediately = false;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create sd context\n");
        return 1;
    }

    double t_load = ms_since(t_load_start);
    auto mem_after_load = get_gpu_mem();
    double rss_after_load = get_rss_mb();
    fprintf(stderr, "Model loaded in %.1f ms\n", t_load);
    fprintf(stderr, "GPU VRAM after load: %.1f MB used / %.1f MB total\n",
            mem_after_load.used_mb(), mem_after_load.total_mb());
    fprintf(stderr, "Process RSS after load: %.1f MB\n", rss_after_load);

    // --- Set up generation params ---
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt          = prompt;
    gen_params.negative_prompt = "";
    gen_params.width           = 512;
    gen_params.height          = 512;
    gen_params.seed            = seed;
    gen_params.batch_count     = 1;
    gen_params.sample_params.sample_steps       = steps;
    gen_params.sample_params.guidance.txt_cfg   = cfg_scale;
    gen_params.sample_params.sample_method = LCM_SAMPLE_METHOD;
    gen_params.sample_params.scheduler = sd_get_default_scheduler(ctx, LCM_SAMPLE_METHOD);
    gen_params.vae_tiling_params.enabled       = true;
    gen_params.vae_tiling_params.tile_size_x   = 32;
    gen_params.vae_tiling_params.tile_size_y   = 32;
    gen_params.vae_tiling_params.target_overlap = 0.5f;

    // --- Warmup run ---
    fprintf(stderr, "\n========== WARMUP RUN (discarded) ==========\n");
    g_profiling = false;
    sd_image_t* warmup = generate_image(ctx, &gen_params);
    if (warmup) { free(warmup[0].data); free(warmup); }

    // --- Profiled run ---
    fprintf(stderr, "\n========== PROFILED RUN ==========\n");
    g_profiling = true;

    auto mem_before = get_gpu_mem();
    double rss_before = get_rss_mb();

    auto t_gen_start = Clock::now();
    sd_image_t* results = generate_image(ctx, &gen_params);
    double t_gen = ms_since(t_gen_start);

    auto mem_after = get_gpu_mem();
    double rss_after = get_rss_mb();

    if (!results) {
        fprintf(stderr, "error: image generation failed\n");
        free_sd_ctx(ctx);
        return 1;
    }

    // Save output
    stbi_write_png("/tmp/cpp_profile_output.png",
                   results[0].width, results[0].height,
                   results[0].channel, results[0].data, 0);

    // --- Summary ---
    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  C++/stable-diffusion.cpp (CUDA, fp16 weights, fp32 compute)\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  Prompt:     \"%s\"\n", prompt);
    fprintf(stderr, "  Resolution: 512x512\n");
    fprintf(stderr, "  Steps:      %d (LCM)\n", steps);
    fprintf(stderr, "  Seed:       %lld\n", (long long)seed);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --- Timing ---\n");
    fprintf(stderr, "  (Per-phase timings above from library logs)\n");
    fprintf(stderr, "  Total generate_image(): %8.1f ms\n", t_gen);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --- GPU Memory (VRAM) ---\n");
    fprintf(stderr, "  After model load: %8.1f MB used\n", mem_after_load.used_mb());
    fprintf(stderr, "  Before inference: %8.1f MB used\n", mem_before.used_mb());
    fprintf(stderr, "  After inference:  %8.1f MB used\n", mem_after.used_mb());
    fprintf(stderr, "  Total VRAM:       %8.1f MB\n", mem_after.total_mb());
    fprintf(stderr, "\n");
    fprintf(stderr, "  --- CPU Memory (RAM) ---\n");
    fprintf(stderr, "  After model load: %8.1f MB RSS\n", rss_after_load);
    fprintf(stderr, "  Before inference: %8.1f MB RSS\n", rss_before);
    fprintf(stderr, "  After inference:  %8.1f MB RSS\n", rss_after);
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  Saved output to /tmp/cpp_profile_output.png\n");

    free(results[0].data);
    free(results);
    free_sd_ctx(ctx);

    return 0;
}
