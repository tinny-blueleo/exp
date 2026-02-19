# stable-diffusion-cpp

Experiment for interfacing with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) via its C API.

## Structure

```
stable-diffusion-cpp/
├── CMakeLists.txt          # Our build (wraps the submodule)
├── main.cpp                # Minimal txt2img CLI using the C API
├── CLAUDE.md               # This file
├── models/                 # Model files (git-ignored)
└── stable-diffusion.cpp/   # Git submodule (leejet/stable-diffusion.cpp)
    └── ggml/               # Nested submodule (ggml backend)
```

## Prerequisites

### Required

- CMake >= 3.12
- C/C++ compiler with C++17 support (GCC, Clang)

### GPU Backends (pick one)

| Backend | System packages | CMake flag |
|---------|----------------|------------|
| **CUDA** (NVIDIA) | `nvidia-cuda-toolkit` | `-DSD_CUDA=ON` |
| **Vulkan** | `libvulkan-dev vulkan-tools` | `-DSD_VULKAN=ON` |
| **ROCm/HIP** (AMD) | ROCm SDK | `-DSD_HIPBLAS=ON` |
| **Metal** (macOS) | Xcode CLI tools | `-DSD_METAL=ON` |
| **OpenCL** | `ocl-icd-opencl-dev` | `-DSD_OPENCL=ON` |

On this machine (NVIDIA GTX 1050 Ti, 4GB VRAM):

```bash
sudo apt install nvidia-cuda-toolkit
```

Verify with `nvcc --version`.

## Build from Scratch

### 1. Clone submodules

```bash
git submodule update --init --recursive
```

### 2. Configure and build

**CPU-only:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=OFF
cmake --build build -j$(nproc)
```

**CUDA (GPU):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=OFF -DSD_CUDA=ON
cmake --build build -j$(nproc)
```

**Vulkan (GPU):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=OFF -DSD_VULKAN=ON
cmake --build build -j$(nproc)
```

> Always `rm -rf build` when switching backends.

### 3. Verify GPU is active

During `cmake -B build ...` look for:

```
-- Use CUDA as backend stable-diffusion     # CUDA
-- Use Vulkan as backend stable-diffusion   # Vulkan
```

If you only see `Including CPU backend`, the GPU backend is **not** enabled.

## Download a Model

Current model: [LCM Dreamshaper v7](https://huggingface.co/Steward/lcm-dreamshaper-v7-gguf) (~2.1GB, SD 1.5-based, fast 4-step inference).

```bash
mkdir -p models
wget -O models/LCM_Dreamshaper_v7-f16.gguf "https://huggingface.co/Steward/lcm-dreamshaper-v7-gguf/resolve/main/LCM_Dreamshaper_v7-f16.gguf"
```

## Usage

Binary is at `./build/sd-experiment`.

### CLI Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model <path>` | `-m` | Path to model (.gguf, .safetensors, .ckpt) | *required* |
| `--diffusion-model <path>` | | Standalone diffusion model | |
| `--vae <path>` | | Standalone VAE model | |
| `--clip_l <path>` | | clip-l text encoder | |
| `--clip_g <path>` | | clip-g text encoder | |
| `--t5xxl <path>` | | T5-XXL text encoder | |
| `--prompt <text>` | `-p` | Generation prompt | `"a lovely cat"` |
| `--negative-prompt <text>` | `-n` | Negative prompt | `""` |
| `--width <int>` | `-W` | Image width | `512` |
| `--height <int>` | `-H` | Image height | `512` |
| `--steps <int>` | | Sampling steps | `20` |
| `--cfg-scale <float>` | | CFG guidance scale | `7.0` |
| `--sampling-method <name>` | | Sampler algorithm | auto |
| `--scheduler <name>` | | Noise scheduler | auto |
| `--vae-tiling` | | Process VAE in tiles (recommended for low VRAM) | off |
| `--vae-conv-direct` | | Use direct convolution for VAE (may be faster) | off |
| `--vae-on-cpu` | | Keep VAE on CPU (saves VRAM, very slow) | off |
| `--clip-on-cpu` | | Keep CLIP on CPU (saves VRAM) | off |
| `--seed <int>` | | RNG seed (-1 for random) | `42` |
| `--threads <int>` | `-t` | Thread count | auto |
| `--output <path>` | `-o` | Output PNG path | `output.png` |

### Sampling Methods

`euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd`, `res_multistep`, `res_2s`

### Schedulers

`discrete`, `karras`, `exponential`, `ays`, `gits`, `smoothstep`, `sgm_uniform`, `simple`, `kl_optimal`, `lcm`, `bong_tangent`

### Examples

**LCM Dreamshaper v7 on GTX 1050 Ti (4GB VRAM) — use `--vae-tiling`:**

```bash
./build/sd-experiment -m models/LCM_Dreamshaper_v7-f16.gguf -p "Taj Mahal in space" --sampling-method lcm --cfg-scale 1.0 --steps 4 --vae-tiling -o output.png
```

**Standard SD 1.5 model (20 steps):**

```bash
./build/sd-experiment -m model.safetensors -p "a photo of an astronaut riding a horse" -n "blurry, bad quality" -W 512 -H 512 --steps 20 --cfg-scale 7.0 --seed 42 -o output.png
```

**Split-model (e.g. Flux, SD3):**

```bash
./build/sd-experiment --diffusion-model unet.safetensors --clip_l clip_l.safetensors --t5xxl t5xxl_fp16.safetensors --vae vae.safetensors -p "prompt"
```

## VRAM Management (GTX 1050 Ti / 4GB)

The f16 model uses ~1970MB VRAM for weights. Without tiling, the VAE decode buffer needs ~1664MB additional VRAM, which exceeds 4GB total and causes OOM.

| Strategy | VAE decode time | Total time (4 LCM steps) | Notes |
|----------|----------------|--------------------------|-------|
| **`--vae-tiling`** | ~10s | **~15s** | Recommended. Tiles VAE into 9 chunks at ~416MB each on GPU |
| `--vae-on-cpu` | ~96s | ~101s | Fallback. Entire VAE runs on CPU |
| No flag | OOM crash | - | VAE needs 1664MB contiguous VRAM |

- **`--vae-tiling`** (recommended): Splits VAE decode into 32x32 tiles with 50% overlap, each needing only ~416MB VRAM. Stays on GPU. ~10x faster than CPU.
- **`--vae-on-cpu`**: Fallback if tiling still OOMs (e.g. larger images). Very slow.
- **`--clip-on-cpu`**: Moves CLIP (~235MB) to RAM. Combine with above for maximum VRAM savings.

## Why is Python/diffusers faster?

Python with PyTorch diffusers can run the same model in <5s total because:

1. **PyTorch uses fp16 compute throughout** — including the VAE. This halves the VAE compute buffer (~832MB vs 1664MB), often fitting in VRAM without tiling.
2. **cuDNN convolution kernels** — PyTorch delegates to NVIDIA's cuDNN library which has highly optimized conv2d implementations. stable-diffusion.cpp uses ggml's own CUDA kernels.
3. **Lazy memory allocation** — PyTorch's CUDA allocator reuses memory between layers and doesn't need to pre-allocate the entire compute graph. ggml allocates the full buffer upfront.
4. **torch.compile / CUDA graphs** — Recent diffusers versions can fuse operations and reduce kernel launch overhead.

## Can we match Python/diffusers speed?

### fp16 VAE compute — not possible yet

ggml hardcodes `GGML_TYPE_F32` as the output type for `ggml_mul_mat()` (in `ggml.c:3184`). Even though model weights are stored in fp16, all intermediate computation runs in fp32, doubling the VRAM needed for compute buffers compared to PyTorch. Fixing this would require changes to ggml core.

### cuDNN — not integrated

ggml uses its own custom CUDA kernels (e.g. `ggml-cuda/conv2d.cu`) rather than NVIDIA's cuDNN library. There are no cmake options to enable cuDNN. This is a major reason PyTorch's conv2d is faster.

### What we can do

- **`--vae-conv-direct`**: Uses direct convolution for VAE. Uses ~60% less VRAM per tile (176MB vs 416MB) but is **~15x slower on GPU** (170s vs 11s). Only useful for extreme memory constraints on CPU; avoid with CUDA.
- **`SD_FAST_SOFTMAX=ON`**: ~1.5x faster softmax on CUDA, but non-deterministic (same seed can produce different images).
- **Self-quantize to q8_0/q5_1**: Reduces model VRAM footprint, potentially avoiding tiling entirely. Use stable-diffusion.cpp's convert mode.

## Notes

- LCM models need `--sampling-method lcm`, `--cfg-scale 1.0`, and `--steps 4` (range 1-8).
- Sampling on GPU takes ~5s for 4 LCM steps at 512x512 on the GTX 1050 Ti.
- No quantized GGUF versions of LCM Dreamshaper v7 exist yet ([only f16 available](https://huggingface.co/Steward/lcm-dreamshaper-v7-gguf)). You can self-quantize using stable-diffusion.cpp's convert mode.
