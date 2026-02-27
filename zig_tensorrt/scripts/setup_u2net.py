#!/usr/bin/env python3
"""Download U2-Net ONNX model and build a TensorRT engine for background removal.

U2-Net is a salient object detection model that produces a foreground mask from
an RGB image. The mask is used as an alpha channel to make backgrounds transparent.

The model comes from the rembg project (same model used by the popular Python
background removal tool).

Usage:
    python scripts/setup_u2net.py [--engine-dir engines/] [--models-dir models/]
"""

import argparse
import os
import sys
import time
import urllib.request

# U2-Net ONNX model from the rembg project's GitHub releases.
# ~176 MB, produces high-quality foreground masks at 320x320 resolution.
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

# TensorRT optimization profile — fixed batch=1, 320x320 input.
# U2-Net input: "input.1" with shape [1, 3, 320, 320]
# U2-Net outputs: "d0" through "d6", each [1, 1, 320, 320]
# We only use "d0" (the primary/finest segmentation mask).
PROFILE = {
    "input.1": ((1, 3, 320, 320), (1, 3, 320, 320), (1, 3, 320, 320)),
}


def download_onnx(models_dir):
    """Download u2net.onnx if not already present."""
    os.makedirs(models_dir, exist_ok=True)
    onnx_path = os.path.join(models_dir, "u2net.onnx")

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"  SKIP: {onnx_path} already exists ({size_mb:.1f} MB)")
        return onnx_path

    print(f"  Downloading u2net.onnx ({U2NET_URL})...")
    urllib.request.urlretrieve(U2NET_URL, onnx_path)
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def build_engine(onnx_path, engine_dir):
    """Convert u2net.onnx to a TensorRT engine with FP16 precision."""
    import tensorrt as trt

    engine_path = os.path.join(engine_dir, "u2net.trt")
    os.makedirs(engine_dir, exist_ok=True)

    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / 1024 / 1024
        print(f"  SKIP: {engine_path} already exists ({size_mb:.1f} MB)")
        return engine_path

    print(f"\nBuilding U2-Net TensorRT engine...")
    start = time.time()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"  Parsing ONNX: {onnx_path}")
    if not parser.parse_from_file(os.path.abspath(onnx_path)):
        for i in range(parser.num_errors):
            print(f"    Error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse {onnx_path}")

    # Print discovered inputs/outputs for verification
    print(f"  Inputs:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape} ({inp.dtype})")
    print(f"  Outputs:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape} ({out.dtype})")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # U2-Net runs well in FP16 — segmentation masks don't need FP32 precision
    config.set_flag(trt.BuilderFlag.FP16)

    # Add optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    for input_name, (min_s, opt_s, max_s) in PROFILE.items():
        found = any(
            network.get_input(i).name == input_name
            for i in range(network.num_inputs)
        )
        if found:
            profile.set_shape(input_name, min_s, opt_s, max_s)
            print(f"    Profile {input_name}: {opt_s}")
    config.add_optimization_profile(profile)

    print(f"  Building engine (this may take a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine for U2-Net")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    elapsed = time.time() - start
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"  Done: {engine_path} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    return engine_path


def main():
    parser = argparse.ArgumentParser(
        description="Download U2-Net and build TensorRT engine for background removal"
    )
    parser.add_argument("--models-dir", default="models", help="ONNX model directory")
    parser.add_argument("--engine-dir", default="engines", help="Output engine directory")
    args = parser.parse_args()

    try:
        import tensorrt
        print(f"Using TensorRT Python API (version {tensorrt.__version__})")
    except ImportError:
        print("ERROR: TensorRT Python package not found.")
        print("Install with: pip install tensorrt")
        sys.exit(1)

    onnx_path = download_onnx(args.models_dir)
    engine_path = build_engine(onnx_path, args.engine_dir)
    print(f"\nU2-Net engine ready: {engine_path}")


if __name__ == "__main__":
    main()
