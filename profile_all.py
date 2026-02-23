#!/usr/bin/env python3
"""Profile all three Stable Diffusion implementations.

Runs TensorRT C++, GGML C++, and Python/diffusers with multiple prompts,
collecting timing, memory, and per-component metrics.

Outputs:
  - Images saved as {prompt_name}_{approach}.png in profile_output/
  - Summary table printed to stdout
"""

import gc
import json
import os
import re
import subprocess
import sys
import time

import psutil
import torch
from diffusers import DiffusionPipeline

# ── Config ──────────────────────────────────────────────────────────────────

PROMPTS = [
    ("taj_mahal", "Taj Mahal in space"),
    ("mario_steak", "Mario eating a steak"),
    ("cyberpunk_cat", "cyberpunk cat with neon lights"),
]

SEED = 42
STEPS = 4
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_output")

TENSORRT_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tensor_rt/build/sd-tensorrt")
TENSORRT_CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_rt")

SD_CPP_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "stable-diffusion-cpp/build/sd-experiment")
SD_CPP_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "stable-diffusion-cpp/models/LCM_Dreamshaper_v7-f16.gguf")

PYTHON_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "python-ml/venv/bin/python")


def get_gpu_mem():
    """Return (allocated_MB, reserved_MB, peak_MB)."""
    return (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
        torch.cuda.max_memory_allocated() / 1024**2,
    )


def get_ram_mb(pid=None):
    p = psutil.Process(pid or os.getpid())
    return p.memory_info().rss / 1024**2


def get_gpu_usage_nvidia_smi():
    """Get current GPU memory usage from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:
        return 0, 0


# ── TensorRT C++ ───────────────────────────────────────────────────────────

def profile_tensorrt(prompt_name, prompt_text, output_path, is_warmup=False):
    """Run TensorRT pipeline and parse its timing output."""
    gpu_before, gpu_total = get_gpu_usage_nvidia_smi()

    cmd = [
        TENSORRT_BIN,
        "--prompt", prompt_text,
        "--seed", str(SEED),
        "--steps", str(STEPS),
        "-o", output_path,
    ]

    t_start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TENSORRT_CWD)
    t_wall = time.perf_counter() - t_start

    gpu_after, _ = get_gpu_usage_nvidia_smi()

    output = result.stdout + result.stderr

    # Parse component timings from output
    text_enc_ms = 0.0
    denoise_ms = 0.0
    vae_ms = 0.0
    total_ms = 0.0
    step_times = []

    for line in output.split("\n"):
        if "Text encoding:" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m: text_enc_ms = float(m.group(1))
        elif "Denoising:" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m: denoise_ms = float(m.group(1))
        elif "VAE decode:" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m: vae_ms = float(m.group(1))
        elif "Total generation:" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m: total_ms = float(m.group(1))
        elif "Step " in line and "ms" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m: step_times.append(float(m.group(1)))

    # Parse engine sizes from output for VRAM estimate
    vram_mb = 0
    for line in output.split("\n"):
        m = re.search(r"Loaded engine:.*\((\d+)\s*MB\)", line)
        if m:
            vram_mb += int(m.group(1))

    return {
        "approach": "TensorRT C++",
        "prompt": prompt_text,
        "wall_time_ms": t_wall * 1000,
        "total_inference_ms": total_ms,
        "text_encode_ms": text_enc_ms,
        "denoise_ms": denoise_ms,
        "vae_decode_ms": vae_ms,
        "step_times_ms": step_times,
        "avg_step_ms": sum(step_times[1:]) / len(step_times[1:]) if len(step_times) > 1 else (sum(step_times) / len(step_times) if step_times else 0),
        "gpu_mem_before_mb": gpu_before,
        "gpu_mem_after_mb": gpu_after,
        "gpu_mem_peak_mb": vram_mb if vram_mb > 0 else gpu_after,
        "output": output_path,
        "raw_output": output,
    }


# ── Stable Diffusion C++ (GGML) ───────────────────────────────────────────

def profile_sd_cpp(prompt_name, prompt_text, output_path, is_warmup=False):
    """Run sd.cpp and parse its timing output."""
    gpu_before, gpu_total = get_gpu_usage_nvidia_smi()

    cmd = [
        SD_CPP_BIN,
        "-m", SD_CPP_MODEL,
        "-p", prompt_text,
        "--sampling-method", "lcm",
        "--cfg-scale", "1.0",
        "--steps", str(STEPS),
        "--seed", str(SEED),
        "-o", output_path,
    ]

    t_start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t_wall = time.perf_counter() - t_start

    gpu_after, _ = get_gpu_usage_nvidia_smi()

    output = result.stdout + result.stderr

    # Parse timings from sd.cpp output
    # Format: "taking 169 ms" or "taking 1.48s"
    text_enc_ms = 0.0
    denoise_ms = 0.0
    vae_ms = 0.0
    total_gen_ms = 0.0

    def parse_time(line):
        """Parse 'taking X ms' or 'taking X.Xs' to milliseconds."""
        m = re.search(r"taking\s+([\d.]+)\s*ms", line)
        if m:
            return float(m.group(1))
        m = re.search(r"taking\s+([\d.]+)\s*s", line)
        if m:
            return float(m.group(1)) * 1000
        # Also handle "completed in X.Xs"
        m = re.search(r"completed in\s+([\d.]+)\s*s", line)
        if m:
            return float(m.group(1)) * 1000
        return 0.0

    for line in output.split("\n"):
        if "get_learned_condition" in line and "taking" in line:
            text_enc_ms = parse_time(line)
        elif "sampling completed" in line and "taking" in line:
            denoise_ms = parse_time(line)
        elif "decode_first_stage" in line and "taking" in line:
            vae_ms = parse_time(line)
        elif "generate_image completed" in line:
            total_gen_ms = parse_time(line)

    total_ms = text_enc_ms + denoise_ms + vae_ms
    if total_ms == 0 and total_gen_ms > 0:
        total_ms = total_gen_ms

    # Parse VRAM usage from sd.cpp output
    vram_mb = 0
    for line in output.split("\n"):
        m = re.search(r"total params memory size\s*=\s*([\d.]+)\s*MB", line)
        if m:
            vram_mb = float(m.group(1))

    return {
        "approach": "SD.cpp (GGML)",
        "prompt": prompt_text,
        "wall_time_ms": t_wall * 1000,
        "total_inference_ms": total_ms if total_ms > 0 else t_wall * 1000,
        "text_encode_ms": text_enc_ms,
        "denoise_ms": denoise_ms,
        "vae_decode_ms": vae_ms,
        "step_times_ms": [],
        "avg_step_ms": denoise_ms / STEPS if denoise_ms > 0 else 0,
        "gpu_mem_before_mb": gpu_before,
        "gpu_mem_after_mb": gpu_after,
        "gpu_mem_peak_mb": vram_mb if vram_mb > 0 else gpu_after,
        "output": output_path,
        "raw_output": output,
    }


# ── Python/diffusers ──────────────────────────────────────────────────────

def profile_python(pipe, prompt_name, prompt_text, output_path, is_warmup=False):
    """Run diffusers pipeline with detailed timing."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    ram_before = get_ram_mb()
    gpu_before, _, _ = get_gpu_mem()
    gpu_smi_before, _ = get_gpu_usage_nvidia_smi()

    generator = torch.Generator("cuda").manual_seed(SEED)

    torch.cuda.synchronize()

    # Time text encoding (run tokenizer + text encoder separately)
    t_enc_start = time.perf_counter()
    text_inputs = pipe.tokenizer(
        prompt_text,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(pipe.device)
    prompt_embeds = pipe.text_encoder(text_input_ids)[0]
    torch.cuda.synchronize()
    t_enc = time.perf_counter() - t_enc_start

    # Time full pipeline (includes re-encoding, but gives accurate total)
    generator2 = torch.Generator("cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    result = pipe(
        prompt_text,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
        generator=generator2,
        output_type="pil",
    )

    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_start

    image = result.images[0]
    image.save(output_path)

    ram_after = get_ram_mb()
    _, _, peak_gpu = get_gpu_mem()
    gpu_smi_after, _ = get_gpu_usage_nvidia_smi()

    return {
        "approach": "Python/diffusers",
        "prompt": prompt_text,
        "wall_time_ms": t_total * 1000,
        "total_inference_ms": t_total * 1000,
        "text_encode_ms": t_enc * 1000,
        "denoise_ms": 0,  # hard to separate in diffusers pipeline
        "vae_decode_ms": 0,
        "step_times_ms": [],
        "avg_step_ms": 0,
        "gpu_mem_before_mb": gpu_smi_before,
        "gpu_mem_after_mb": gpu_smi_after,
        "gpu_mem_peak_mb": peak_gpu,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
        "output": output_path,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def fmt_ms(ms):
    if ms == 0: return "  -  "
    if ms < 1000: return f"{ms:>7.1f}ms"
    return f"{ms/1000:>7.2f}s "


def print_results_table(all_results):
    print("\n" + "=" * 120)
    print("  PROFILING RESULTS — LCM Dreamshaper v7, 512x512, 4 steps, seed 42")
    print("=" * 120)

    # Group by prompt
    by_prompt = {}
    for r in all_results:
        key = r["prompt"]
        by_prompt.setdefault(key, []).append(r)

    # Per-prompt comparison
    for prompt, results in by_prompt.items():
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  {'Approach':<22} {'Total':>10} {'TextEnc':>10} {'Denoise':>10} {'VAE Dec':>10} {'Avg Step':>10} {'GPU VRAM':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in sorted(results, key=lambda x: x["total_inference_ms"]):
            gpu_str = f"{r.get('gpu_mem_peak_mb', 0):>6.0f}MB"
            print(f"  {r['approach']:<22} {fmt_ms(r['total_inference_ms'])} {fmt_ms(r['text_encode_ms'])} "
                  f"{fmt_ms(r['denoise_ms'])} {fmt_ms(r['vae_decode_ms'])} "
                  f"{fmt_ms(r['avg_step_ms'])} {gpu_str}")

    # Summary: average across prompts
    print(f"\n{'=' * 120}")
    print(f"  AVERAGES ACROSS ALL PROMPTS")
    print(f"{'=' * 120}")

    by_approach = {}
    for r in all_results:
        by_approach.setdefault(r["approach"], []).append(r)

    print(f"\n  {'Approach':<22} {'Avg Total':>10} {'Avg TextEnc':>12} {'Avg Denoise':>12} {'Avg VAE':>10} {'Avg/Step':>10} {'Peak GPU':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for approach in ["TensorRT C++", "SD.cpp (GGML)", "Python/diffusers"]:
        if approach not in by_approach:
            continue
        results = by_approach[approach]
        avg_total = sum(r["total_inference_ms"] for r in results) / len(results)
        avg_enc = sum(r["text_encode_ms"] for r in results) / len(results)
        avg_denoise = sum(r["denoise_ms"] for r in results) / len(results)
        avg_vae = sum(r["vae_decode_ms"] for r in results) / len(results)
        avg_step = sum(r["avg_step_ms"] for r in results) / len(results)
        peak_gpu = max(r.get("gpu_mem_peak_mb", 0) for r in results)

        gpu_str = f"{peak_gpu:>6.0f}MB"
        print(f"  {approach:<22} {fmt_ms(avg_total)} {fmt_ms(avg_enc):>12} "
              f"{fmt_ms(avg_denoise):>12} {fmt_ms(avg_vae)} {fmt_ms(avg_step)} {gpu_str}")

    # Web service perspective
    print(f"\n{'=' * 120}")
    print(f"  WEB SERVICE PERSPECTIVE")
    print(f"  (Startup = model/engine loading, Per-Request = inference only)")
    print(f"{'=' * 120}")

    for approach in ["TensorRT C++", "SD.cpp (GGML)", "Python/diffusers"]:
        if approach not in by_approach:
            continue
        results = by_approach[approach]
        avg_total = sum(r["total_inference_ms"] for r in results) / len(results)
        print(f"\n  {approach}:")
        if approach == "TensorRT C++":
            print(f"    Startup (load 3 engines + tokenizer): ~3-5s (one-time)")
            print(f"    Per-request inference:                 {fmt_ms(avg_total).strip()}")
            print(f"    Peak throughput:                       ~{60000/avg_total:.0f} req/min")
        elif approach == "SD.cpp (GGML)":
            print(f"    Startup (load GGUF model):             ~2-3s (one-time)")
            print(f"    Per-request inference:                 {fmt_ms(avg_total).strip()}")
            print(f"    Peak throughput:                       ~{60000/avg_total:.0f} req/min")
        else:
            print(f"    Startup (load pipeline + GPU warmup):  ~5-10s (one-time)")
            print(f"    Per-request inference:                 {fmt_ms(avg_total).strip()}")
            print(f"    Peak throughput:                       ~{60000/avg_total:.0f} req/min")

    # Speedup comparison
    print(f"\n{'=' * 120}")
    print(f"  SPEEDUP vs TensorRT C++")
    print(f"{'=' * 120}")

    if "TensorRT C++" in by_approach:
        trt_avg = sum(r["total_inference_ms"] for r in by_approach["TensorRT C++"]) / len(by_approach["TensorRT C++"])
        for approach in ["TensorRT C++", "SD.cpp (GGML)", "Python/diffusers"]:
            if approach not in by_approach:
                continue
            results = by_approach[approach]
            avg = sum(r["total_inference_ms"] for r in results) / len(results)
            speedup = avg / trt_avg
            bar = "#" * int(speedup * 10)
            print(f"  {approach:<22} {fmt_ms(avg).strip():>10}  {speedup:>5.1f}x  {bar}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    # ── 1. TensorRT C++ ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PROFILING: TensorRT C++")
    print("=" * 60)

    # Warmup run
    print("\n  Warmup run...")
    warmup_path = os.path.join(OUTPUT_DIR, "_warmup_trt.png")
    profile_tensorrt("warmup", "warmup test", warmup_path, is_warmup=True)

    for prompt_name, prompt_text in PROMPTS:
        out_path = os.path.join(OUTPUT_DIR, f"{prompt_name}_tensorrt.png")
        print(f"\n  Running: \"{prompt_text}\"")
        r = profile_tensorrt(prompt_name, prompt_text, out_path)
        all_results.append(r)
        print(f"    Total: {r['total_inference_ms']:.1f}ms "
              f"(text={r['text_encode_ms']:.1f}ms denoise={r['denoise_ms']:.1f}ms "
              f"vae={r['vae_decode_ms']:.1f}ms)")

    # ── 2. Stable Diffusion C++ (GGML) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  PROFILING: SD.cpp (GGML)")
    print("=" * 60)

    # Warmup
    print("\n  Warmup run...")
    warmup_path = os.path.join(OUTPUT_DIR, "_warmup_sdcpp.png")
    profile_sd_cpp("warmup", "warmup test", warmup_path, is_warmup=True)

    for prompt_name, prompt_text in PROMPTS:
        out_path = os.path.join(OUTPUT_DIR, f"{prompt_name}_sdcpp.png")
        print(f"\n  Running: \"{prompt_text}\"")
        r = profile_sd_cpp(prompt_name, prompt_text, out_path)
        all_results.append(r)
        print(f"    Total: {r['total_inference_ms']:.1f}ms "
              f"(text={r['text_encode_ms']:.1f}ms denoise={r['denoise_ms']:.1f}ms "
              f"vae={r['vae_decode_ms']:.1f}ms)")

    # ── 3. Python/diffusers ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PROFILING: Python/diffusers")
    print("=" * 60)

    print("\n  Loading pipeline...")
    t_load = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.enable_model_cpu_offload()
    t_load = time.perf_counter() - t_load
    print(f"  Pipeline loaded in {t_load:.1f}s")

    # Warmup
    print("\n  Warmup run...")
    warmup_path = os.path.join(OUTPUT_DIR, "_warmup_python.png")
    profile_python(pipe, "warmup", "warmup test", warmup_path, is_warmup=True)

    for prompt_name, prompt_text in PROMPTS:
        out_path = os.path.join(OUTPUT_DIR, f"{prompt_name}_python.png")
        print(f"\n  Running: \"{prompt_text}\"")
        r = profile_python(pipe, prompt_name, prompt_text, out_path)
        all_results.append(r)
        print(f"    Total: {r['total_inference_ms']:.1f}ms")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    # ── Results ────────────────────────────────────────────────────────
    print_results_table(all_results)

    # Save raw data
    json_path = os.path.join(OUTPUT_DIR, "profile_results.json")
    json_safe = []
    for r in all_results:
        d = {k: v for k, v in r.items() if k != "raw_output"}
        json_safe.append(d)
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nRaw data saved to {json_path}")


if __name__ == "__main__":
    main()
