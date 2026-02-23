#!/usr/bin/env python3
"""Compare our C++ scheduler against the diffusers reference step-by-step."""

import torch
import numpy as np
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float32
)

sched = LCMScheduler.from_config(pipe.scheduler.config)
sched.set_timesteps(4, original_inference_steps=50)

print("=== Scheduler Config ===")
for k, v in pipe.scheduler.config.items():
    if not k.startswith("_"):
        print(f"  {k}: {v}")

print(f"\n=== Timesteps ===")
print(f"  diffusers: {sched.timesteps.tolist()}")

# Check what our C++ produces
# C++ formula: original_inference_steps=50, c=1000/50=20
# lcm_origin_timesteps = [(i+1)*20 - 1 for i in range(50)] = [19, 39, ..., 999]
# skipping_step = 50 // 4 = 12
# indices: start from 49, step -12: [49, 37, 25, 13]
# timesteps: [999, 759, 519, 279]
cpp_timesteps = []
c = 1000 // 50
lcm_origin = [(i + 1) * c - 1 for i in range(50)]
skip = 50 // 4  # = 12
indices = list(range(49, -1, -skip))[:4]
cpp_timesteps = [lcm_origin[i] for i in indices]
print(f"  C++ (ours): {cpp_timesteps}")

print(f"\n=== Alpha cumprod at each timestep ===")
for t in sched.timesteps.tolist():
    print(f"  t={t}: alpha_cumprod={sched.alphas_cumprod[t]:.6f}")

print(f"\n=== Step-by-step comparison ===")
print(f"  final_alpha_cumprod (set_alpha_to_one={sched.config.set_alpha_to_one}): {sched.final_alpha_cumprod:.6f}")

# Simulate our C++ scheduler logic
print(f"\n=== C++ scheduler simulation ===")
# init same as diffusers
np.random.seed(42)
latents = np.random.randn(1, 4, 64, 64).astype(np.float32)

# Get reference embeddings
tokenizer = pipe.tokenizer
tokens = tokenizer("Taj Mahal in space", padding="max_length", max_length=77,
                   truncation=True, return_tensors="pt")
with torch.no_grad():
    embeddings = pipe.text_encoder(tokens.input_ids)[0]

# Run reference
sched_ref = LCMScheduler.from_config(pipe.scheduler.config)
sched_ref.set_timesteps(4, original_inference_steps=50)

latents_ref = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(42))
latents_ref = latents_ref * sched_ref.init_noise_sigma

print(f"\nReference (diffusers) step details:")
for i, t in enumerate(sched_ref.timesteps):
    with torch.no_grad():
        noise_pred = pipe.unet(latents_ref, t, encoder_hidden_states=embeddings).sample

    # Get internal scheduler state
    step_idx = sched_ref._step_index if hasattr(sched_ref, '_step_index') else i

    result = sched_ref.step(noise_pred, t, latents_ref)
    latents_ref = result.prev_sample

    # Try to get predicted_original_sample
    x0 = result.pred_original_sample if hasattr(result, 'pred_original_sample') and result.pred_original_sample is not None else None

    print(f"  Step {i}: t={t.item()}")
    print(f"    noise_pred: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]")
    if x0 is not None:
        print(f"    x0_pred:    [{x0.min():.4f}, {x0.max():.4f}]")
    print(f"    latents:    [{latents_ref.min():.4f}, {latents_ref.max():.4f}]")

# Now print what the C++ scheduler does differently
print(f"\n=== Key differences to check ===")
print(f"1. Timesteps: diffusers={sched.timesteps.tolist()} vs C++={cpp_timesteps}")

# Check prev_timestep logic
print(f"\n2. Previous timesteps (for alpha_t_prev):")
for i, t in enumerate(sched.timesteps.tolist()):
    if i + 1 < len(sched.timesteps):
        t_prev = sched.timesteps[i + 1].item()
        print(f"   Step {i}: t={t} -> t_prev={t_prev}, alpha_t_prev={sched.alphas_cumprod[t_prev]:.6f}")
    else:
        # Last step: what does diffusers use?
        if sched.config.set_alpha_to_one:
            print(f"   Step {i}: t={t} -> t_prev=FINAL (set_alpha_to_one=True), alpha_t_prev=1.0")
        else:
            print(f"   Step {i}: t={t} -> t_prev=FINAL, alpha_t_prev={sched.alphas_cumprod[0]:.6f}")

print(f"\n   C++ uses alpha_t_prev=1.0 for the final step (matching set_alpha_to_one=True)")

# Check if diffusers LCMScheduler uses denoised_epsilon or just x0
print(f"\n3. LCM step formula check:")
print(f"   prediction_type: {sched.config.prediction_type}")
print(f"   clip_sample: {sched.config.clip_sample}")
print(f"   clip_sample_range: {sched.config.clip_sample_range}")
