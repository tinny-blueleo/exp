"""Profile LCM Dreamshaper v7 inference with PyTorch/diffusers.

Loads model once, does a warmup run, then profiles a second run with
timing and GPU/CPU memory stats.
"""

import gc
import os
import time

import psutil
import torch
from diffusers import DiffusionPipeline


PROMPT = "Taj Mahal in space"
SEED = 42
STEPS = 4
WIDTH = 512
HEIGHT = 512


def get_gpu_mem():
    """Return (allocated_MB, reserved_MB, peak_allocated_MB)."""
    return (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
        torch.cuda.max_memory_allocated() / 1024**2,
    )


def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def run_inference(pipe, label):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    ram_before = get_ram_mb()
    gpu_alloc_before, gpu_reserved_before, _ = get_gpu_mem()

    generator = torch.Generator("cuda").manual_seed(SEED)

    # Time total inference
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    result = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
        generator=generator,
        output_type="pil",
    )

    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_start

    image = result.images[0]

    ram_after = get_ram_mb()
    gpu_alloc_after, gpu_reserved_after, peak_gpu = get_gpu_mem()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Prompt:     \"{PROMPT}\"")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  Steps:      {STEPS} (LCM)")
    print(f"  Seed:       {SEED}")
    print()
    print(f"  --- Timing ---")
    print(f"  Total inference:  {t_total*1000:>8.1f} ms")
    print()
    print(f"  --- GPU Memory (VRAM) ---")
    print(f"  Before inference: {gpu_alloc_before:>8.1f} MB allocated, {gpu_reserved_before:.1f} MB reserved")
    print(f"  After inference:  {gpu_alloc_after:>8.1f} MB allocated, {gpu_reserved_after:.1f} MB reserved")
    print(f"  Peak allocated:   {peak_gpu:>8.1f} MB")
    print()
    print(f"  --- CPU Memory (RAM) ---")
    print(f"  Process RSS before: {ram_before:>8.1f} MB")
    print(f"  Process RSS after:  {ram_after:>8.1f} MB")
    print(f"{'='*60}")

    return image


def main():
    print("Loading model (fp16, model_cpu_offload, no safety checker)...")
    t_load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.enable_model_cpu_offload()
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.2f}s")
    print(f"Process RSS after load: {get_ram_mb():.1f} MB")

    # Warmup run
    print("\n--- Warmup run (discarded) ---")
    run_inference(pipe, "WARMUP")

    # Profiled run
    print("\n--- Profiled run ---")
    image = run_inference(pipe, "Python/diffusers (fp16, model_cpu_offload)")

    image.save("/tmp/python_profile_output.png")
    print(f"\nSaved output to /tmp/python_profile_output.png")


if __name__ == "__main__":
    main()
