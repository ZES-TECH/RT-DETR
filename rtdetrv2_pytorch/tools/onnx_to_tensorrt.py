import argparse
import os
import sys

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Helper function to build the TensorRT engine from an ONNX file.
# In this example, we assume the ONNX model has two inputs:
#   1) "images" with shape [N, 3, 640, 640] (batch x channels x height x width)
#   2) "orig_target_sizes" with shape [N, 2].
# If you want to handle truly dynamic shapes (varying batch, resolution, etc.),
# you should create an optimization profile for each dynamic dimension.
# Here we show a simple approach setting a single shape.

def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine for a 2-input ONNX model.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX file.")
    parser.add_argument("--engine", type=str, default="model.engine", help="Output path for serialized engine.")
    parser.add_argument("--workspace", type=int, default=2, help="Max workspace size in GB (default=2).")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 if supported.")
    parser.add_argument("--batch", type=int, default=1, help="Max batch size.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    return args

def build_engine_from_onnx(
    onnx_file_path: str,
    engine_file_path: str,
    max_workspace_size_gb: int = 2,
    fp16: bool = False,
    max_batch_size: int = 1,
    verbose: bool = False,
):
    """
    Builds a TensorRT engine from an ONNX file and serializes it to "engine_file_path".

    Args:
        onnx_file_path (str): Path to the ONNX model file.
        engine_file_path (str): Where to save the serialized TRT engine.
        max_workspace_size_gb (int): GPU builder workspace in GB.
        fp16 (bool): Whether to enable FP16 if supported.
        max_batch_size (int): The batch size for setting input shapes.
        verbose (bool): Whether to enable verbose logs.
    """
    # 1) Create Builder / Network / Parser
    if verbose:
        trt_logger = trt.Logger(trt.Logger.VERBOSE)
    else:
        trt_logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(trt_logger)
    network_creation_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_creation_flags)
    parser = trt.OnnxParser(network, trt_logger)
    for i in range(parser.num_errors):
        print(parser.get_error(i))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    max_workspace_size_gb * (1024 ** 3)
    )


    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[build_engine] FP16 mode enabled.")
        else:
            print("[build_engine] Warning: FP16 not supported on this platform. Using FP32.")

    # 2) Parse the ONNX file
    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise ValueError("[build_engine] Failed to parse the ONNX file.")

    # 3) Create an optimization profile for dynamic inputs (images, orig_target_sizes)
    # Assuming "images" has shape [N, 3, 640, 640], "orig_target_sizes" has shape [N, 2].
    # We'll set min=opt=max for a fixed shape approach.

    profile = builder.create_optimization_profile()

    # For "images": set the shape to (max_batch_size, 3, 640, 640)
    # We'll assume a fixed 640 resolution, but you can expand to truly dynamic dims.
    images_min = (1, 3, 640, 640)
    images_opt = (1, 3, 640, 640)
    images_max = (max_batch_size, 3, 640, 640)

    profile.set_shape("images", images_min, images_opt, images_max)

    # For "orig_target_sizes": shape [N, 2]
    sizes_min = (1, 2)
    sizes_opt = (1, 2)
    sizes_max = (max_batch_size, 2)

    profile.set_shape("orig_target_sizes", sizes_min, sizes_opt, sizes_max)

    config.add_optimization_profile(profile)

    # 4) Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        raise RuntimeError("Failed to build serialized engine.")
    
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        if not engine:
            raise RuntimeError("[build_engine] Failed to build the engine.")

        # 5) Serialize engine to file
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
    print(f"[build_engine] Engine saved to {engine_file_path}")


def main():
    args = parse_args()

    if not os.path.isfile(args.onnx):
        print(f"[Error] ONNX file not found: {args.onnx}")
        sys.exit(1)

    build_engine_from_onnx(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        max_workspace_size_gb=args.workspace,
        fp16=args.fp16,
        max_batch_size=args.batch,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
