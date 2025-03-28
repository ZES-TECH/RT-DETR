import os
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import time
import cv2
import numpy as np
import onnxruntime as ort
import onnx
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

# Initialize transformation once
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

def load_onnx_model(onnx_path):
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return session

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(Image.fromarray(image))[None]
    return img_tensor.data.numpy()

def infer_onnx(session, input_data, orig_size):
    orig_size = np.array([orig_size[0], orig_size[1]])[None]
    start_time = time.time()
    output = session.run(None, {'images': input_data, "orig_target_sizes": orig_size})
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to ms
    return output, latency

def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    utilization = nvmlDeviceGetUtilizationRates(handle)
    return utilization.gpu  # GPU utilization in percentage

def process_image(image, session):
    h, w, _ = image.shape
    input_data = preprocess_image(image)
    _, latency = infer_onnx(session, input_data, (w, h))
    return latency

def process_video(video_path, session):
    cap = cv2.VideoCapture(video_path)
    total_time = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        latency = process_image(frame, session)
        total_time += latency
        frame_count += 1
    
    cap.release()
    avg_latency = total_time / frame_count if frame_count > 0 else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0
    return avg_latency, fps

def process_source(source, session):
    total_time = 0
    frame_count = 0
    
    if os.path.isfile(source):
        if source.endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(source)
            avg_latency = process_image(image, session)
            fps = 1000 / avg_latency if avg_latency > 0 else 0
        else:
            avg_latency, fps = process_video(source, session)
    elif os.path.isdir(source):
        images = [cv2.imread(os.path.join(source, img)) for img in os.listdir(source) if img.endswith(('.png', '.jpg', '.jpeg'))]
        for img in images:
            if img is None:
                continue
            latency = process_image(img, session)
            total_time += latency
            frame_count += 1
        avg_latency = total_time / frame_count if frame_count > 0 else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0
    else:
        raise ValueError("Invalid source path")
    
    gpu_util = get_gpu_utilization()
    return avg_latency, fps, gpu_util

def compute_flops(onnx_path):
    model = onnx.load(onnx_path)
    
    total_flops = 0
    for node in model.graph.node:
        if node.op_type in ["Conv", "Gemm", "MatMul"]:  # Identify major compute-heavy layers
            input_names = node.input
            input_shapes = {}
            
            # Retrieve tensor shapes from the model's initializers
            for initializer in model.graph.initializer:
                if initializer.name in input_names:
                    dims = list(initializer.dims)
                    input_shapes[initializer.name] = dims
            
            # Estimate FLOPs based on the operation type
            if node.op_type == "Conv" and len(input_shapes) > 0:
                # Convolution FLOPs = 2 * (input_channels * output_channels * kernel_size * height * width)
                input_shape = list(input_shapes.values())[0]
                if len(input_shape) == 4:  # Format: (C, H, W, N) or (N, C, H, W)
                    flops = 2 * np.prod(input_shape)
                    total_flops += flops

            elif node.op_type in ["Gemm", "MatMul"] and len(input_shapes) > 0:
                # Fully connected layer FLOPs = 2 * (input_dim * output_dim)
                input_shape = list(input_shapes.values())[0]
                if len(input_shape) >= 2:
                    flops = 2 * np.prod(input_shape)
                    total_flops += flops

    return total_flops / 1e9  # Convert to GFLOPs

def main():
    parser = argparse.ArgumentParser(description="ONNX GPU Inference with Speed Analysis")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--source", type=str, required=True, help="Path to image, video, or folder")
    args = parser.parse_args()
    
    session = load_onnx_model(args.onnx_path)
    print("Model loaded")
    avg_latency, fps, gpu_util = process_source(args.source, session)
    print("Processed Source")
    gflops = compute_flops(args.onnx_path)
    
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"GPU Utilization: {gpu_util}%")
    print(f"GFLOPs: {gflops:.2f}")

if __name__ == "__main__":
    main()
