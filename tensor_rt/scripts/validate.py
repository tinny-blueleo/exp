#!/usr/bin/env python3
"""Validate pipeline components against diffusers reference."""

import torch
import numpy as np
from diffusers import DiffusionPipeline, LCMScheduler

print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float32
)

# 1. Check scheduler config and timesteps
sched = LCMScheduler.from_config(pipe.scheduler.config)
sched.set_timesteps(4, original_inference_steps=50)
print(f"\nScheduler config: {pipe.scheduler.config}")
print(f"Timesteps (4 steps): {sched.timesteps.tolist()}")
print(f"Alphas cumprod[0]: {sched.alphas_cumprod[0]:.6f}")
print(f"Alphas cumprod[249]: {sched.alphas_cumprod[249]:.6f}")
print(f"Alphas cumprod[499]: {sched.alphas_cumprod[499]:.6f}")
print(f"Alphas cumprod[749]: {sched.alphas_cumprod[749]:.6f}")
print(f"Alphas cumprod[999]: {sched.alphas_cumprod[999]:.6f}")

# 2. Check text encoder
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

tokens = tokenizer(
    "Taj Mahal in space",
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt",
)
print(f"\nToken IDs (first 10): {tokens.input_ids[0][:10].tolist()}")

with torch.no_grad():
    enc_out = text_encoder(tokens.input_ids)
    embeddings = enc_out[0]  # last_hidden_state

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings min={embeddings.min():.4f} max={embeddings.max():.4f} mean={embeddings.mean():.6f}")
print(f"Embeddings[0,0,:5]: {embeddings[0,0,:5].tolist()}")
print(f"Embeddings[0,1,:5]: {embeddings[0,1,:5].tolist()}")

# 3. Check what enc_out has
print(f"\nText encoder output keys: {enc_out.keys() if hasattr(enc_out, 'keys') else 'tuple of ' + str(len(enc_out))}")

# 4. Run actual generation with scheduler step-by-step
print("\n--- Reference generation ---")
sched2 = LCMScheduler.from_config(pipe.scheduler.config)
sched2.set_timesteps(4, original_inference_steps=50)

generator = torch.Generator().manual_seed(42)
latents = torch.randn(1, 4, 64, 64, generator=generator)
# Scale by init_noise_sigma
latents = latents * sched2.init_noise_sigma
print(f"Init latents: min={latents.min():.4f} max={latents.max():.4f}")
print(f"init_noise_sigma: {sched2.init_noise_sigma}")

for i, t in enumerate(sched2.timesteps):
    with torch.no_grad():
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=embeddings).sample
    print(f"Step {i+1} t={t}: noise min={noise_pred.min():.4f} max={noise_pred.max():.4f}")
    latents = sched2.step(noise_pred, t, latents).prev_sample
    print(f"  latents: min={latents.min():.4f} max={latents.max():.4f}")

# VAE decode
with torch.no_grad():
    latents_scaled = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents_scaled).sample
print(f"\nDecoded image: min={image.min():.4f} max={image.max():.4f}")
