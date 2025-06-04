import argparse
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.function import validate
from lib.models import get_net
from lib.utils.utils import create_logger, select_device

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')

    parser.add_argument('--config',
                    help='model configuration',
                    type=str,
                    default='base')
    parser.add_argument('--outDir',
                        help='output directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--weights', type=str, default='', help='model.pth path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args

def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir = create_logger(
        cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')
    # device = select_device(logger, 'cpu')

    model = get_net(cfg)
    print("finish build model")
    

    # det_idx_range = [str(i) for i in range(0,25)]
    model_dict = model.state_dict()
    checkpoint_file = args.weights
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint_dict = torch.load(checkpoint_file)
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    # model.half()
    model.gr = 1.0
    model.nc = 1
    print('bulid model finished')

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    print('load data finished')

    epoch = 0 #special for test
    da_segment_results,ll_segment_results,detect_results, times = validate(
        epoch,cfg, valid_loader, valid_dataset, model,
        logger, device
    )
    msg =   'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1])
    logger.info(msg)


if __name__ == '__main__':
    main()
    