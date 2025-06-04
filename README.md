# TriLiteNet



<img width="1161" alt="Ảnh màn hình 2025-04-27 lúc 05 00 57" src="https://github.com/user-attachments/assets/b57d9590-b57d-475d-a93f-cfef960f9669" />

## Docker

To ease testing of this model, use the docker image

```shell
docker compose build
docker compose run --rm dev
```


## Requirement

This codebase has been developed with python version 3.8, PyTorch 1.8.0 and torchvision 0.9.0

```setup
pip install -r requirements.txt
```



## Pre-trained Model
You can get the pre-trained model from <a href="https://drive.google.com/drive/folders/1wLZqemCxxzwiFeFUGY1zMaqcKoQLHFyK?usp=sharing">here</a>.


## Dataset
- Download the images from [images](https://bdd-data.berkeley.edu/).
- Download the annotations of detection from [det_annotations](https://drive.google.com/file/d/1Ge-R8NTxG1eqd4zbryFo-1Uonuh0Nxyl/view?usp=sharing). 
- Download the annotations of drivable area segmentation from [da_seg_annotations](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view?usp=sharing). 
- Download the annotations of lane line segmentation from [ll_seg_annotations](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view?usp=sharing). 

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_seg_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `lib/config/default.py`.

### Custom Datasets

For custom datasets, use the tools in /tools to create a dataset in BDD100k format

#### Detection Annotations

```shell
python dataset/detection.py  --image_dir '{path to image directory}' --output_dir '{path to detection annotations directory}' --append
```

You can also use view the object detections of a dataset by running the command

```shell
python dataset/view.py  --image_dir '{path to image directory}' --output_dir '{path to detection annotations directory}'
```

#### Detection Annotations

```shell
python dataset/lanes.py  --image_dir '{path to image directory}' --output_dir '{path to lane annotations directory}' --append
```

#### Detection Annotations

```shell
python dataset/drivable.py  --image_dir '{path to image directory}' --output_dir '{path to drivable area annotations directory}' --append
```

#### Utils

You can find other useful scripts such as:

* To create a sample of a dataset

```shell
./dataset/sample.sh -d DATASET_DIR -s SAMPLE_DIR -t TRAIN_SAMPLES -v VAL_SAMPLES
```

* To create a train/val split from a set of images

```shell
./dataset/split.sh -d SOURCE_DIR -s SAMPLE_DIR -t TRAIN_SAMPLES -v VAL_SAMPLES
```

* To merge datasets

```shell
./dataset/merge.sh -d SOURCE_DIR -d SOURCE_DIR -m MERGED_DIR
```

## Training

```shell
python tools/train.py --config '{nano/small/medium/large}'
```

## Transfer learning

```shell
python tools/transfer.py --config '{nano/small/medium/large}'
```

## Evaluation

```shell
python tools/test.py --config '{tiny/small/base}' --weights 'weights/{tiny/small/base}.pth'
```
## Demo

```shell
python tools/demo.py --config '{tiny/small/base}' --weights 'weights/{tiny/small/base}.pth' --source 'inference/videos/1.mp4' or 'inference/images'
```

## Conversion to ONNX

```shell
python tools/convert_to_onnx.py --config '{tiny/small/base}' --weights 'weights/{tiny/small/base}.pth' --name 'base.onnx'
```

## Demo ONNX model

```shell
python tools/demo_onnx.py --weights 'weights/{tiny/small/base}.onnx' --source 'inference/images'
```


## Acknowledgements

This work would not have been possible without the valuable contributions of the following authors.


* [YOLOP](https://github.com/hustvl/YOLOP)
* [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
* [TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus)
* [Partial Class Activation Attention for Semantic Segmentation](https://github.com/lsa1997/PCAA)
* [ESPNet](https://github.com/sacmehta/ESPNet)

## Citation

```BibTeX
@ARTICLE{trilitenet,
  author={Che, Quang-Huy and Lam, Duc-Khai},
  journal={IEEE Access}, 
  title={TriLiteNet: Lightweight Model for Multi-Task Visual Perception}, 
  year={2025},
  volume={13},
  number={},
  pages={50152-50166},
  doi={10.1109/ACCESS.2025.3552088}}

```

