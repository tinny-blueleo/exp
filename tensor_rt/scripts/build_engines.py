#!/usr/bin/env python3
"""Build TensorRT engines from ONNX models.

Converts exported ONNX files into optimized TensorRT FP16 engines.
Uses fixed shapes (batch=1, 512x512) to minimize VRAM during build.

Usage:
    python scripts/build_engines.py [--onnx-dir models/] [--engine-dir engines/]

Alternatively, use trtexec directly:
    trtexec --onnx=models/text_encoder.onnx --saveEngine=engines/text_encoder.trt --fp16
"""

import argparse
import os
import subprocess
import sys
import time


def build_with_trtexec(onnx_path, engine_path, extra_args=None):
    """Build a TensorRT engine using trtexec CLI."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--memPoolSize=workspace:1024MiB",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"trtexec failed for {onnx_path}")
    return True


def build_with_python_api(onnx_path, engine_path):
    """Build a TensorRT engine using the Python API."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"  Parsing ONNX: {onnx_path}")
    onnx_abs = os.path.abspath(onnx_path)
    if not parser.parse_from_file(onnx_abs):
        for i in range(parser.num_errors):
            print(f"    Error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)

    print(f"  Building engine (this may take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"Failed to build engine for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)


def build_engine(name, onnx_dir, engine_dir, use_trtexec):
    onnx_path = os.path.join(onnx_dir, f"{name}.onnx")
    engine_path = os.path.join(engine_dir, f"{name}.trt")

    if not os.path.exists(onnx_path):
        print(f"  SKIP: {onnx_path} not found")
        return False

    if os.path.exists(engine_path):
        print(f"  SKIP: {engine_path} already exists (delete to rebuild)")
        return True

    print(f"\nBuilding {name} engine...")
    start = time.time()

    if use_trtexec:
        build_with_trtexec(onnx_path, engine_path)
    else:
        build_with_python_api(onnx_path, engine_path)

    elapsed = time.time() - start
    size = os.path.getsize(engine_path)
    print(f"  Done: {engine_path} ({size / 1024 / 1024:.1f} MB) in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX")
    parser.add_argument("--onnx-dir", default="models", help="ONNX model directory")
    parser.add_argument(
        "--engine-dir", default="engines", help="Output engine directory"
    )
    parser.add_argument(
        "--trtexec",
        action="store_true",
        help="Use trtexec CLI instead of Python API",
    )
    args = parser.parse_args()

    os.makedirs(args.engine_dir, exist_ok=True)

    # Determine build method
    use_trtexec = args.trtexec
    if not use_trtexec:
        try:
            import tensorrt

            print(f"Using TensorRT Python API (version {tensorrt.__version__})")
        except ImportError:
            print("TensorRT Python package not found, falling back to trtexec")
            use_trtexec = True

    if use_trtexec:
        if subprocess.run(["which", "trtexec"], capture_output=True).returncode != 0:
            print("ERROR: trtexec not found. Install tensorrt-dev or tensorrt-libs.")
            sys.exit(1)
        print("Using trtexec CLI")

    models = ["text_encoder", "unet", "vae_decoder"]
    success = True
    for name in models:
        if not build_engine(name, args.onnx_dir, args.engine_dir, use_trtexec):
            success = False

    if success:
        print("\nAll engines built successfully!")
        print(f"Files in {args.engine_dir}/:")
        for f in sorted(os.listdir(args.engine_dir)):
            if f.endswith(".trt"):
                size = os.path.getsize(os.path.join(args.engine_dir, f))
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
    else:
        print("\nSome engines failed to build.")
        sys.exit(1)


if __name__ == "__main__":
    main()
