# zig_tensorrt — GPU-Accelerated Inference Library in Zig

A Zig library and CLI for running TensorRT-accelerated ML inference pipelines. Currently implements text-to-image generation using LCM Dreamshaper v7 (Stable Diffusion 1.5) with optional U2-Net background removal.

## Project Layout

```
zig_tensorrt/
├── build.zig                    Build config — exposes "sd_tensorrt" public module
├── src/
│   ├── lib.zig                  Library root — public API (ImageFromPromptParams, generateImageForPrompt)
│   ├── main.zig                 CLI executable — parses args, imports sd_tensorrt module, writes PNG
│   ├── engine.zig               Shared TensorRT engine wrapper (@cImport of trt_wrapper.h)
│   └── text_to_image/           Text-to-image pipeline (LCM Dreamshaper v7)
│       ├── inference.zig        InferenceModels — loads all engines once, owns everything
│       ├── pipeline.zig         3-phase orchestration: CLIP → UNet → VAE
│       ├── scheduler.zig        Pure Zig LCM noise scheduler
│       ├── tokenizer.zig        Pure Zig CLIP BPE tokenizer
│       └── background_removal.zig  U2-Net post-processing for transparent backgrounds
├── trt_wrapper/                 C++ bridge (shared by all pipelines)
│   ├── trt_wrapper.h/cpp        C ABI layer — opaque TrtEngine handle
│   ├── trt_engine.h/cpp         TensorRT C++ engine class
│   └── stb_impl.cpp             stb_image_write compiled as C++
├── include/
│   └── stb_image_write.h        Vendored header for PNG output
├── scripts/
│   ├── setup_u2net.py           Downloads U2-Net ONNX and builds TRT engine
│   └── (uses tensor_rt/scripts/ for SD model setup)
├── data/                        Symlink → ../tensor_rt/data (BPE vocab)
├── engines/                     TRT engine files (gitignored, built by scripts)
└── models/                      ONNX intermediates (gitignored, downloaded by scripts)
```

## Build & Run

```bash
# Prerequisites: Zig 0.16+, NVIDIA GPU, TensorRT 10+, CUDA runtime
# Engines must be pre-built via Python scripts (see Setup below)

zig build
./zig-out/bin/sd-tensorrt-zig --prompt "ninja cat" --seed 42 -o output.png
./zig-out/bin/sd-tensorrt-zig --prompt "ninja cat" --seed 42 --transparent -o output_rgba.png
```

## Setup (one-time, requires Python + TensorRT Python package)

```bash
# 1. Build SD engines (from tensor_rt project)
cd ../tensor_rt
python scripts/download_onnx.py
python scripts/build_engines.py

# 2. Build U2-Net engine (for --transparent)
cd ../zig_tensorrt
python scripts/setup_u2net.py
```

## Using as a Library

Other Zig projects import the `sd_tensorrt` module:

```zig
const sd = @import("sd_tensorrt");

// Load models once at startup (~2-3 seconds)
var models = try sd.TextToImageModels.init(allocator, .{
    .engine_dir = "engines",
    .vocab_path = "data/bpe_simple_vocab_16e6.txt",
    .enable_transparency = true,
});
defer models.deinit();

// Generate images (~450ms each)
const pixels = try sd.generateImageForPrompt(&models, .{
    .prompt = "ninja cat",
    .seed = 42,
    .steps = 4,
    .transparent = true,
});
defer allocator.free(pixels);
// pixels is []u8: RGB (512*512*3) or RGBA (512*512*4)
```

## Architecture Notes

- **Why C++ wrapper?** TensorRT is C++ only (virtual classes, no C ABI). Zig needs a thin C wrapper (`trt_wrapper/`) to call it via `@cImport`.
- **Why not LoRA?** LCM Dreamshaper v7 is a full distilled checkpoint (~3.3GB), NOT a LoRA. Lineage: SD 1.5 → Dreamshaper v7 (full fine-tune) → LCM distillation.
- **TRT engines are GPU-specific.** A `.trt` file built on one GPU architecture won't work on another. Engines must be built on the target machine.
- **Thread safety:** `generate()` / `generateTransparent()` are NOT thread-safe. For concurrent server use, add a mutex or engine pool (see TODO in inference.zig).
- **Pipeline folder structure** supports future pipelines (image-to-image, image-to-text, text-to-audio) under `src/<pipeline_name>/`.
