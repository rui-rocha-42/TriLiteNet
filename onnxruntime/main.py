#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import onnxruntime as ort

#import torch
import os
import glob
from pathlib import Path
from tqdm import tqdm
import time

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

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

class MultiTaskTriLite():
    def __init__(self, model_path, confThreshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open('onnxruntime/classes', 'r').readlines()))
        print(self.classes)
        self.num_class = len(self.classes)

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, session_options=so, providers=providers, provider_options=provider_options)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.confThreshold = confThreshold
        self.nmsThreshold = 0.5
        anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
        self.na = len(anchors[0]) // 2
        self.no = len(self.classes) + 5
        self.stride = [8, 16, 32]
        self.nl = len(self.stride)
        self.anchors = np.asarray(anchors, dtype=np.float32).reshape(3, 3, 1, 1, 2)
        self.generate_grid()
        self.input_type = self.session.get_inputs()[0].type
        self.use_fp16 = self.input_type == "tensor(float16)"

        print("Performing warmup pass...")
        dummy_input = np.random.randn(1, 3, self.input_width, self.input_height).astype(np.float16)
        if self.use_fp16:
            dummy_input = dummy_input.astype(np.float16)  # Convert to float16 if required
        self.session.run(None, {self.input_name: dummy_input})
        print("Warmup pass completed.")

    def generate_grid(self):
        self.grid = []
        for i in range(self.nl):
            h, w = int(self.input_height / self.stride[i]), int(self.input_width / self.stride[i])
            self.grid.append(self._make_grid(w, h))
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), 2).reshape(1, 1, ny, nx, 2).astype(np.float32)

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), 0, 0.7, (0, 255, 0), thickness=2)
        return frame
    def detect(self, frame):
        image_width, image_height = frame.shape[1], frame.shape[0]
        ratioh = image_height / self.input_height
        ratiow = image_width / self.input_width

        # Pre process:Resize, BGR->RGB, float32 cast
        input_image = cv2.resize(frame, dsize=(self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32)
        if self.use_fp16:
            input_image = input_image.astype(np.float16)
        input_image = input_image / 255.0

        start_time = time.time()
        # Inference
        results = self.session.run(None, {self.input_name: input_image})
        end_time = time.time()

        total_time = end_time - start_time

        # Object Detection
        det_out = results[0].squeeze(axis=0)

        boxes, confidences, classIds = [], [], []

        for i in range(det_out.shape[0]):
            if det_out[i, 4] * np.max(det_out[i, 5:]) < self.confThreshold:
                continue

            class_id = np.argmax(det_out[i, 5:])
            cx, cy, w, h = det_out[i, :4]
            x = int((cx - 0.5*w) * ratiow)
            y = int((cy - 0.5*h) * ratioh)
            width = int(w * ratiow)
            height = int(h* ratioh)

            boxes.append([x, y, width, height])
            classIds.append(class_id)
            confidences.append(det_out[i, 4] * np.max(det_out[i, 5:]))
        start_time = time.time()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        end_time = time.time()
        total_time_nms = end_time - start_time
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

        # Drivable Area Segmentation
        drivable_area = np.squeeze(results[1], axis=0)
        mask = np.argmax(drivable_area, axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame[mask==1] = [0, 255, 0]
        # Lane Line
        lane_line = np.squeeze(results[2])
        mask = np.where(lane_line > 0.5, 1, 0).astype(np.uint8)
        mask = np.argmax(mask, axis=0).astype(np.uint8)
        print(mask.shape, image_width, image_height)
        print(lane_line.shape)
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame[mask==1] = [255, 0, 0]
        return frame, (total_time, total_time_nms)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='onnx/base.onnx', help="model path")
    parser.add_argument("--source", type=str, default='inference/images', help="image path")
    parser.add_argument("--save-dir", type=str, default='inference/output', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    args = parser.parse_args()

    net = MultiTaskTriLite(args.modelpath, confThreshold=args.confThreshold)

    p = str(Path(args.source))  # os-agnostic
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
    
    total_time = 0.0
    total_time_nms = 0.0
    for i, img_path in tqdm(enumerate(images), total=len(images)):    
        srcimg = cv2.imread(img_path)
        srcimg, (inf_time, nms_time) = net.detect(srcimg)
        total_time += inf_time
        total_time_nms += nms_time
        save_path = str(args.save_dir +'/'+ Path(img_path).name)
        cv2.imwrite(save_path, srcimg)

    fps = len(images) / total_time
    fps_nms = len(images) / total_time_nms
    print(f"FPS Test: {fps:.2f} frames per second")
    print(f"FPS NMS: {fps_nms:.2f} frames per second")
