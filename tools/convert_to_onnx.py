import os
import torch
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


from lib.models import get_net
from lib.config import cfg, update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
import argparse
import onnx
import onnxsim

import torch
import torch.nn as nn

class ONNXCompatibleNMS(torch.nn.Module):
    def __init__(self, conf_thres=0.25, iou_thres=0.45, max_output_boxes_per_class=100):
        super(ONNXCompatibleNMS, self).__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_output_boxes_per_class = max_output_boxes_per_class

    def forward(self, boxes, scores):
        # Filter boxes based on confidence threshold
        mask = scores > self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]

        # Apply ONNX-compatible NMS
        indices = torch.ops.onnx.NonMaxSuppression(
            boxes,
            scores.unsqueeze(0),  # ONNX expects scores to be 2D
            self.max_output_boxes_per_class,
            self.iou_thres,
            self.conf_thres,
        )

        # Gather filtered boxes and scores
        filtered_boxes = torch.index_select(boxes, 0, indices)
        filtered_scores = torch.index_select(scores, 0, indices)

        return filtered_boxes, filtered_scores

class ModelWithNMS(torch.nn.Module):
    def __init__(self, original_model, conf_thres=0.25, iou_thres=0.45):
        super(ModelWithNMS, self).__init__()
        self.original_model = original_model
        self.nms = ONNXCompatibleNMS(conf_thres, iou_thres)

    def forward(self, x):
        # Get raw outputs from the original model
        det_out, da_seg_out, ll_seg_out = self.original_model(x)
        print(type(det_out), det_out[0].shape if det_out is tuple else len(det_out))

        # Extract bounding box coordinates and scores
        boxes = det_out[..., :4]  # [x1, y1, x2, y2]
        scores = det_out[..., 4]  # Confidence scores

        # Apply NMS
        filtered_boxes, filtered_scores = self.nms(boxes, scores)

        return filtered_boxes, filtered_scores, da_seg_out, ll_seg_out

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--pth_model", type=str, required=True, help="Path to the .pth model file")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to save the ONNX model")
    parser.add_argument("--image_size", type=int, nargs=2, default=(640, 640), help="Input image size (height, width)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for ONNX export")
    return parser.parse_args()

def main(cfg,opt):

    logger, _ = create_logger(
        cfg, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
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
    height = width = opt.img_size
    dummy_input = torch.randn(1, 3, height, width).to(device)
    if half:
        dummy_input = dummy_input.half()  # Convert input tensor to FP16 if model is in FP16

    # Export to ONNX
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if opt.dynamic else None
    save_path = f"{opt.save_dir}/{opt.name}"
    print(f"Exporting model to ONNX format at {save_path}...")

    onnx_program = torch.onnx.export(
        model,
        dummy_input,
        output_names=["det_out", "wtv1", "wtv2", "wtv3", "da_seg_out", "ll_seg_out"], # Output names for the ONNX model
        opset_version=18,  # Specify the ONNX opset version
        verbose=True,
        dynamo=True
    )
    onnx_program.optimize()
    onnx_program.save(save_path)

    # Load the ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)  # check onnx model
    #print(onnx.helper.printable_graph(onnx_model.graph))  # print

    #onnx_model, check = onnxsim.simplify(onnx_model)
    #assert check, 'assert check failed'

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
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
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