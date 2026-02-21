#!/usr/bin/env python3
"""Export LCM Dreamshaper v7 components to ONNX format.

Exports three separate ONNX models:
  - text_encoder.onnx  (CLIP text encoder)
  - unet.onnx          (UNet denoiser)
  - vae_decoder.onnx   (VAE decoder)

All exported in FP32. TensorRT handles FP16 conversion during engine build.

Usage:
    python scripts/export_onnx.py [--output-dir models/]
"""

import argparse
import os
import sys

import torch
from diffusers import DiffusionPipeline


def export_text_encoder(pipe, output_dir):
    print("Exporting text encoder...")
    text_encoder = pipe.text_encoder.float().cpu()
    text_encoder.eval()

    dummy_input = torch.zeros(1, 77, dtype=torch.int32, device="cpu")

    output_path = os.path.join(output_dir, "text_encoder.onnx")
    torch.onnx.export(
        text_encoder,
        (dummy_input,),
        output_path,
        opset_version=17,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes=None,
    )
    print(f"  Saved: {output_path}")


def export_unet(pipe, output_dir):
    print("Exporting UNet (this may take a while)...")
    unet = pipe.unet.float().cpu()
    unet.eval()

    dummy_latents = torch.randn(1, 4, 64, 64, dtype=torch.float32, device="cpu")
    dummy_timestep = torch.tensor([999], dtype=torch.long, device="cpu")
    dummy_encoder_hidden = torch.randn(1, 77, 768, dtype=torch.float32, device="cpu")

    output_path = os.path.join(output_dir, "unet.onnx")
    torch.onnx.export(
        unet,
        (dummy_latents, dummy_timestep, dummy_encoder_hidden),
        output_path,
        opset_version=17,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        dynamic_axes=None,
    )
    print(f"  Saved: {output_path}")


def export_vae_decoder(pipe, output_dir):
    print("Exporting VAE decoder...")
    vae = pipe.vae.float().cpu()
    vae.eval()

    class VAEDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            latents = latents / self.vae.config.scaling_factor
            return self.vae.decode(latents, return_dict=False)[0]

    decoder = VAEDecoder(vae).float().cpu()

    dummy_latents = torch.randn(1, 4, 64, 64, dtype=torch.float32, device="cpu")

    output_path = os.path.join(output_dir, "vae_decoder.onnx")
    torch.onnx.export(
        decoder,
        (dummy_latents,),
        output_path,
        opset_version=17,
        input_names=["latents"],
        output_names=["image"],
        dynamic_axes=None,
    )
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export LCM Dreamshaper to ONNX")
    parser.add_argument(
        "--output-dir", default="models", help="Output directory for ONNX files"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading LCM Dreamshaper v7 pipeline (FP32 for CPU export)...")
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float32,
    )

    export_text_encoder(pipe, args.output_dir)
    export_unet(pipe, args.output_dir)
    export_vae_decoder(pipe, args.output_dir)

    print("\nAll ONNX models exported successfully.")
    print(f"Files in {args.output_dir}/:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
