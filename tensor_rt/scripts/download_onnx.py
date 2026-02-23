#!/usr/bin/env python3
"""Download pre-exported ONNX models from HuggingFace.

Downloads text_encoder, unet, and vae_decoder ONNX files from
SimianLuo/LCM_Dreamshaper_v7 instead of re-exporting from PyTorch.

Usage:
    python scripts/download_onnx.py [--output-dir models/]
"""

import argparse
import os

from huggingface_hub import hf_hub_download

REPO_ID = "SimianLuo/LCM_Dreamshaper_v7"

# Files to download: (subfolder, filename, required)
FILES = [
    ("text_encoder", "model.onnx", True),
    ("unet", "model.onnx", True),
    ("unet", "model.onnx_data", True),
    ("vae_decoder", "model.onnx", True),
]


def main():
    parser = argparse.ArgumentParser(description="Download ONNX models from HuggingFace")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for subfolder, filename, required in FILES:
        out_subdir = os.path.join(args.output_dir, subfolder)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, filename)

        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  SKIP: {out_path} already exists ({size_mb:.1f} MB)")
            continue

        print(f"  Downloading {subfolder}/{filename}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            subfolder=subfolder,
            filename=filename,
            local_dir=args.output_dir,
        )
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"    Saved: {out_path} ({size_mb:.1f} MB)")

    print("\nAll ONNX models downloaded.")
    for subfolder in ["text_encoder", "unet", "vae_decoder"]:
        onnx_path = os.path.join(args.output_dir, subfolder, "model.onnx")
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / 1024 / 1024
            print(f"  {onnx_path}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
