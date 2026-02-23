# TensorRT Stable Diffusion Pipeline

Pure C++ text-to-image inference using NVIDIA TensorRT. Generates 512x512 images from text prompts using the LCM Dreamshaper v7 model.

## Architecture

Three-phase pipeline, each using a separate TensorRT engine:
1. **CLIP Text Encoder** (FP32) — tokenize prompt → embeddings [1, 77, 768]
2. **UNet Diffusion** (FP16) — 4 LCM denoising steps on latent space [1, 4, 64, 64]
3. **VAE Decoder** (FP16) — decode latents to RGB image [1, 3, 512, 512]

Text encoder uses FP32 because FP16 clips embedding values and degrades image quality.

Target: RTX 3050 Laptop (4GB VRAM). If all engines don't fit in VRAM simultaneously, the pipeline automatically switches to sequential loading/unloading.

## Prerequisites

```bash
# TensorRT (provides NvInfer.h, libnvinfer, libnvonnxparsers)
sudo apt install tensorrt-dev tensorrt-libs

# Python deps (for one-time ONNX download + engine build only)
# Uses ../python-ml/venv which already has diffusers, torch, huggingface_hub
```

## Setup & Run

```bash
# 1. Download CLIP vocabulary
mkdir -p data
wget -O data/bpe_simple_vocab_16e6.txt \
  "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz" \
  && gunzip data/bpe_simple_vocab_16e6.txt.gz

# 2. Download ONNX models from HuggingFace (one-time)
../python-ml/venv/bin/python scripts/download_onnx.py --output-dir models/

# 3. Build TensorRT engines (one-time, ~5-10 min)
../python-ml/venv/bin/python scripts/build_engines.py --onnx-dir models/ --engine-dir engines/

# 4. Build C++
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 5. Generate
./build/sd-tensorrt --prompt "Taj Mahal in space" --steps 4 --seed 42 -o output.png
```

## CLI Options

```
--prompt TEXT   Text prompt (default: "Taj Mahal in space")
--seed N        Random seed (default: 42)
--steps N       LCM denoising steps (default: 4)
-o FILE         Output PNG path (default: output.png)
--engine-dir D  TensorRT engine directory (default: engines/)
--vocab FILE    BPE vocab file (default: data/bpe_simple_vocab_16e6.txt)
```

## Key Files

- `scripts/download_onnx.py` — Download ONNX models from HuggingFace (one-time)
- `scripts/build_engines.py` — ONNX → TensorRT engine compilation (one-time)
- `src/engine.h/cpp` — TensorRT engine wrapper (load, infer, memory management)
- `src/tokenizer.h/cpp` — CLIP BPE tokenizer (pure C++)
- `src/scheduler.h/cpp` — LCM noise scheduler (pure C++)
- `src/pipeline.h/cpp` — Orchestrates the 3-phase pipeline
- `src/main.cpp` — CLI entry point, PNG output via stb_image_write

## Build Dependencies

- TensorRT 10.x (via `tensorrt-dev`)
- CUDA Runtime (via `nvidia-cuda-toolkit`)
- stb_image_write (vendored header-only)
