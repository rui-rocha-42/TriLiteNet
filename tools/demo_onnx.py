import argparse
import os
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import sys
import torch
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.core.general import non_max_suppression
from lib.core.postprocess import morphological_process, connect_lane

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)

def detect(opt):

    # Clean output
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    
    # Initialize ONNX Runtime session
    print(f"Loading ONNX model from {opt.weights}...")
    session = ort.InferenceSession(opt.weights, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Execution Providers:", session.get_providers())
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type
    output_names = [output.name for output in session.get_outputs()]

    # Print ONNX model outputs for debugging
    print("ONNX Model Outputs:")
    for output in session.get_outputs():
        print(f"Name: {output.name}, Shape: {output.shape}, Type: {output.type}")

    # Load class names
    names = ['person', 'rider', 'car', 'bus', 'truck', 
            'bike', 'motor', 'tl_green', 'tl_red', 
            'tl_yellow', 'tl_none', 'traffic sign', 'train']


    # Determine if the model expects float16 inputs
    use_fp16 = input_type == "tensor(float16)"

    p = str(Path(opt.source))  # os-agnostic
    p = os.path.abspath(p)  # absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception('ERROR: %s does not exist' % p)

    images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
    assert len(images) > 0, 'No images found in %s. Supported formats are:\nimages: %s' % \
                            (p, img_formats)
    
    total_inference_time = 0
    total_pipeline_time = 0
    num_frames = len(images)

    # Run inference
    for i, img_path in tqdm(enumerate(images), total=len(images)):
        t_pre = time_synchronized()
        img = cv2.imread(img_path)
        # Preprocess the image
        # convert to RGB
        height, width, _ = img.shape
        img_rgb = img[:, :, ::-1].copy()

        # resize & normalize
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,640,640)
        if use_fp16:
            img = img.astype(np.float16)  # Convert to float16 if required

        # Inference
        t_infer = time_synchronized()
        outputs = session.run(["det_out", "da_seg_out", "ll_seg_out"], {input_name: img})
        t_infer_end = time_synchronized()
        total_inference_time += (t_infer_end - t_infer)

        det_out, da_seg_out, ll_seg_out = outputs
        det_out = torch.from_numpy(det_out).float()


        # Post-process detection
        det_pred = non_max_suppression(det_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        det = det_pred[0]

        if det.shape[0] == 0:
            print("no bounding boxes detected.")
        else:
            # scale coords to original size.
            det[:, 0] -= dw
            det[:, 1] -= dh
            det[:, 2] -= dw
            det[:, 3] -= dh
            det[:, :4] /= r

            print(f"detect {det.shape[0]} bounding boxes.")

            img_det = img_rgb[:, :, ::-1].copy()
            for i in range(det.shape[0]):
                x1, y1, x2, y2, conf, label = det[i]
                x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
                img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

        # select da & ll segment area.
        da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

        da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

        color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
        color_area[da_seg_mask == 1] = [0, 255, 0]
        color_area[ll_seg_mask == 1] = [255, 0, 0]
        color_seg = color_area

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        color_mask = np.mean(color_seg, 2)
        img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
        img_merge = img_merge[:, :, ::-1]

        # merge: resize to original size
        img_merge[color_mask != 0] = \
            img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        img_merge = img_merge.astype(np.uint8)
        img_merge = cv2.resize(img_merge, (width, height),
                            interpolation=cv2.INTER_LINEAR)
        for i in range(det.shape[0]):
            x1, y1, x2, y2, conf, label = det[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
            # Add the label and confidence score
            label_det_pred = f'{names[int(label)]} {conf:.2f}'
            img_merge = cv2.putText(img_merge, label_det_pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)  # Add label text

        # da: resize to original size
        da_seg_mask = da_seg_mask * 255
        da_seg_mask = da_seg_mask.astype(np.uint8)
        da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                                interpolation=cv2.INTER_LINEAR)

        # ll: resize to original size
        ll_seg_mask = ll_seg_mask * 255
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        
        t_post_end = time_synchronized()
        total_pipeline_time += (t_post_end - t_pre)
        
        save_path = str(opt.save_dir +'/'+ Path(img_path).name)

        cv2.imwrite(save_path,img_merge)

    fps_inference = num_frames / total_inference_time
    fps_pipeline = num_frames / total_pipeline_time
    print(f"Inference FPS: {fps_inference:.2f}")
    print(f"Pipeline FPS (Preprocessing + Inference + Post-processing): {fps_pipeline:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to ONNX model file')
    parser.add_argument('--source', type=str, default='inference/videos/1.mp4', help='Source file/folder')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='Directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)