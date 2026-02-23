#!/usr/bin/env python3
"""Build TensorRT engines from HuggingFace ONNX models.

Compiles ONNX files (downloaded via download_onnx.py) into optimized TensorRT
engines. Text encoder is built in FP32 (FP16 clips embedding values), UNet and
VAE are built in FP16.

Usage:
    python scripts/build_engines.py [--onnx-dir models/] [--engine-dir engines/]
"""

import argparse
import os
import sys
import time

# Fixed optimization profiles for batch=1, 512x512 (64x64 latents).
PROFILES = {
    "text_encoder": {
        "input_ids": ((1, 77), (1, 77), (1, 77)),
    },
    "unet": {
        "sample": ((1, 4, 64, 64), (1, 4, 64, 64), (1, 4, 64, 64)),
        "timestep": ((1,), (1,), (1,)),
        "encoder_hidden_states": ((1, 77, 768), (1, 77, 768), (1, 77, 768)),
        "timestep_cond": ((1, 256), (1, 256), (1, 256)),
    },
    "vae_decoder": {
        "latent_sample": ((1, 4, 64, 64), (1, 4, 64, 64), (1, 4, 64, 64)),
    },
}


def build_engine(name, onnx_dir, engine_dir):
    import tensorrt as trt

    onnx_path = os.path.join(onnx_dir, name, "model.onnx")
    engine_path = os.path.join(engine_dir, f"{name}.trt")

    if not os.path.exists(onnx_path):
        print(f"  SKIP: {onnx_path} not found")
        return False

    if os.path.exists(engine_path):
        print(f"  SKIP: {engine_path} already exists (delete to rebuild)")
        return True

    print(f"\nBuilding {name} engine...")
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

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Text encoder must stay FP32 — FP16 clips embedding values and degrades quality
    if name != "text_encoder":
        config.set_flag(trt.BuilderFlag.FP16)

    # Add optimization profile for dynamic shapes
    if name in PROFILES:
        profile = builder.create_optimization_profile()
        for input_name, (min_s, opt_s, max_s) in PROFILES[name].items():
            found = any(
                network.get_input(i).name == input_name
                for i in range(network.num_inputs)
            )
            if found:
                profile.set_shape(input_name, min_s, opt_s, max_s)
                print(f"    {input_name}: {opt_s}")
        config.add_optimization_profile(profile)

    print(f"  Building engine (this may take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"Failed to build engine for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    elapsed = time.time() - start
    size = os.path.getsize(engine_path)
    print(f"  Done: {engine_path} ({size / 1024 / 1024:.1f} MB) in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX")
    parser.add_argument("--onnx-dir", default="models", help="ONNX model directory")
    parser.add_argument("--engine-dir", default="engines", help="Output engine directory")
    args = parser.parse_args()

    os.makedirs(args.engine_dir, exist_ok=True)

    try:
        import tensorrt
        print(f"Using TensorRT Python API (version {tensorrt.__version__})")
    except ImportError:
        print("ERROR: TensorRT Python package not found.")
        print("Install with: pip install tensorrt")
        sys.exit(1)

    models = ["text_encoder", "unet", "vae_decoder"]
    success = all(build_engine(name, args.onnx_dir, args.engine_dir) for name in models)

    if success:
        print("\nAll engines built successfully!")
        for f in sorted(os.listdir(args.engine_dir)):
            if f.endswith(".trt"):
                size = os.path.getsize(os.path.join(args.engine_dir, f))
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
    else:
        print("\nSome engines failed to build.")
        sys.exit(1)


if __name__ == "__main__":
    main()
