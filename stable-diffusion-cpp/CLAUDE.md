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
wget -O models/LCM_Dreamshaper_v7-f16.gguf \
    "https://huggingface.co/Steward/lcm-dreamshaper-v7-gguf/resolve/main/LCM_Dreamshaper_v7-f16.gguf"
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
| `--vae-on-cpu` | | Keep VAE on CPU (saves VRAM) | off |
| `--clip-on-cpu` | | Keep CLIP on CPU (saves VRAM) | off |
| `--seed <int>` | | RNG seed (-1 for random) | `42` |
| `--threads <int>` | `-t` | Thread count | auto |
| `--output <path>` | `-o` | Output PNG path | `output.png` |

### Sampling Methods

`euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd`, `res_multistep`, `res_2s`

### Schedulers

`discrete`, `karras`, `exponential`, `ays`, `gits`, `smoothstep`, `sgm_uniform`, `simple`, `kl_optimal`, `lcm`, `bong_tangent`

### Examples

**LCM Dreamshaper v7 on GTX 1050 Ti (4GB VRAM) — requires `--vae-on-cpu`:**

```bash
./build/sd-experiment -m models/LCM_Dreamshaper_v7-f16.gguf -p "Taj Mahal in space" --sampling-method lcm --cfg-scale 1.0 --steps 4 --vae-on-cpu -o output.png
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

The f16 model uses ~1970MB VRAM for weights alone. The VAE decode buffer needs an additional ~1664MB, which exceeds 4GB total. Solutions:

- **`--vae-on-cpu`** (recommended): Runs VAE decode on CPU. Sampling stays on GPU (~5s for 4 LCM steps), VAE decode is slower (~96s on CPU) but avoids OOM. This is required for f16 models on 4GB cards.
- **`--clip-on-cpu`**: Moves CLIP text encoder (~235MB) to RAM. Frees VRAM for larger models.
- Both flags can be combined for maximum VRAM savings.
- Using quantized models (Q4_0, Q5_0) reduces weight VRAM and may allow VAE to stay on GPU.

## Notes

- `SD_FAST_SOFTMAX=ON` gives ~1.5x faster softmax on CUDA but makes results non-deterministic (same seed can produce different images).
- LCM models need `--sampling-method lcm`, `--cfg-scale 1.0`, and `--steps 4` (range 1-8).
- Sampling on GPU takes ~5s for 4 LCM steps at 512x512 on the GTX 1050 Ti.
