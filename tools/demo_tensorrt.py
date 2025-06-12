import onnxruntime as ort
import numpy as np
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run ONNX model inference using TensorRT.")
parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model file.")
args = parser.parse_args()

# Your ONNX model path
model_path = args.model

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"ONNX model file not found at {model_path}")

# Example dummy input data
dummy_input_shape = (1, 3, 320, 320)  # Replace with your model's expected input shape
dummy_input_dtype = np.float16
dummy_input = np.random.rand(*dummy_input_shape).astype(dummy_input_dtype)

# 1. Prioritize TensorRTExecutionProvider, then CUDAExecutionProvider, then CPUExecutionProvider
providers = [
    'TensorrtExecutionProvider',
    'CUDAExecutionProvider',  # Fallback to CUDA if TensorRT can't handle some ops
    'CPUExecutionProvider'    # Fallback to CPU if neither GPU provider works
]

# 2. Configure TensorRT EP options
provider_options = [
    {
        "trt_fp16_enable": True,  # Enable FP16 inference
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB workspace
        "trt_engine_cache_enable": True,  # Enable engine caching
        "trt_engine_cache_path": "./trt_cache"  # Path to store cached engines
    },
    {},  # Empty options for CUDAExecutionProvider
    {}   # Empty options for CPUExecutionProvider
]

# Ensure the TensorRT cache directory exists
os.makedirs("./trt_cache", exist_ok=True)

try:
    # Create the InferenceSession
    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(
        model_path,
        providers=providers,
        provider_options=provider_options
    )
    print(f"ONNX Runtime session created successfully.")
    print(f"Actual providers used: {session.get_providers()}")

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input name: {input_name}, Output name: {output_name}")

    # Warmup pass
    print("Performing warmup pass...")
    session.run([output_name], {input_name: dummy_input})
    print("Warmup pass completed.")

    # FPS test
    print("Performing FPS test...")
    num_iterations = 100  # Number of iterations for the FPS test
    import time
    start_time = time.time()
    for _ in range(num_iterations):
        session.run([output_name], {input_name: dummy_input})
    end_time = time.time()

    # Calculate FPS
    total_time = end_time - start_time
    fps = num_iterations / total_time
    print(f"FPS Test: {fps:.2f} frames per second")

    # Run inference
    print("Running inference...")
    output = session.run([output_name], {input_name: dummy_input})[0]

    # Print results
    print(f"Inference successful. Output shape: {output.shape}")
    print(f"Output data (first 5 values): {output.flatten()[:5]}")

except Exception as e:
    print(f"Failed to create ONNX Runtime session with TensorRTExecutionProvider: {e}")
    print("Please ensure you have onnxruntime-gpu installed and compatible versions of CUDA, cuDNN, and TensorRT are set up correctly.")
    print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")