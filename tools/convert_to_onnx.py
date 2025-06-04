import os
import torch
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


from lib.models import get_net
from lib.config import cfg
from lib.utils.utils import create_logger, select_device
import argparse
import onnx
import onnxsim

def main(cfg,opt):

    logger, _ = create_logger(
        cfg, 'demo')

    device = select_device(logger,opt.device)
    os.makedirs(opt.save_dir, exist_ok=True)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    if half:
        model.half()  # to FP16

    model.eval() 


    # Create a dummy input tensor
    height, width = opt.img_size
    dummy_input = torch.randn(1, 3, height, width).to(device)
    if half:
        dummy_input = dummy_input.half()  # Convert input tensor to FP16 if model is in FP16

    # Export to ONNX
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},  # Dynamic input dimensions
        "det_out": {0: "batch_size"},                        # Dynamic batch size for detection
        "da_seg_out": {0: "batch_size", 2: "height", 3: "width"},  # Dynamic dimensions for drivable area segmentation
        "ll_seg_out": {0: "batch_size", 2: "height", 3: "width"},  # Dynamic dimensions for lane segmentation
    } if opt.dynamic else None
    save_path = f"{opt.save_dir}/{opt.name}"
    print(f"Exporting model to ONNX format at {save_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        output_names=["det_out", "wtv1", "wtv2", "wtv3", "da_seg_out", "ll_seg_out"], # Output names for the ONNX model
        opset_version=18,  # Specify the ONNX opset version
        verbose=True,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,  # Enable constant folding for optimization
    )

    # Load the ONNX model
    onnx_model = onnx.load(save_path)
    print(f"IR version: {onnx_model.ir_version}")
    onnx.checker.check_model(onnx_model)  # check onnx model
    #print(onnx.helper.printable_graph(onnx_model.graph))  # print

    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'assert check failed'

    # Remove unwanted outputs
    desired_outputs = ["det_out", "da_seg_out", "ll_seg_out"]
    outputs_to_keep = [output for output in onnx_model.graph.output if output.name in desired_outputs]

    # Clear the existing outputs and add only the desired ones
    onnx_model.graph.output.clear()
    onnx_model.graph.output.extend(outputs_to_keep)

    # Save the modified ONNX model
    onnx.save(onnx_model, save_path)

    print(f"Model successfully exported to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/base.pth', help='model.pth path(s)')
    parser.add_argument('--config', help='model configuration', type=str, default='base')
    parser.add_argument('--img-size', type=int, nargs=2, default=(640,640), help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='onnx', help='directory to save onnx model')
    parser.add_argument("--name", type=str, default='base.onnx', help="ONNX model name")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for ONNX export")
    opt = parser.parse_args()
    # update_config(cfg, opt)
    cfg.config = opt.config
    with torch.no_grad():
        main(cfg,opt)