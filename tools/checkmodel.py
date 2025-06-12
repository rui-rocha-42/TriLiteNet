import onnx
import sys

def check_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Check for int64 tensors
    int64_found = False
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.INT64:  # Check if data type is int64
            print(f"Tensor '{initializer.name}' has data type int64.")
            int64_found = True

    if not int64_found:
        print("No tensors with data type int64 found in the ONNX model.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python checkmodel.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    check_model(model_path)