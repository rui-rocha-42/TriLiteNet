import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from torch.nn import Upsample
from lib.models.common import Encoder, DetectHead, Detect, SegmentHead, UpSimpleBlock, sc_ch_dict
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from typing import Tuple



def TriLiteNet(model_cfg):
    TriLiteNet = [
        [2, 4, 5],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
        [ -1, Encoder, [model_cfg]],   #0         /2
        [ -1, DetectHead, [model_cfg]],   #1         
        [ -1, Detect,  [6, [[4,12,7,19,11,28], [17,40,25,58,38,89], [62,136,88,206,124,412]], [model_cfg['chanels'][3], model_cfg['chanels'][3], model_cfg['chanels'][3]]]], #2
        [ 0, SegmentHead, [model_cfg]], #3
        [ 3, UpSimpleBlock, [model_cfg['chanels'][0], 2]],  #4
        [ 3, UpSimpleBlock, [model_cfg['chanels'][0], 2]],   #5
    ]
    return TriLiteNet



class MultiTaskModel(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MultiTaskModel, self).__init__()
        layers, save= [], []
        self.nc = 1 # number of classes
        self.detector_index = -1
        self.seg_da_idx = block_cfg[0][1]
        self.seg_ll_idx = block_cfg[0][2]
        

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        #print("model names", self.names)
        # self.names = ['person', 'rider', 'car', 'bus', 'truck', 
        #     'bike', 'motor', 'tl_green', 'tl_red', 
        #     'tl_yellow', 'tl_none', 'traffic sign', 'train']
        
        self.names = ['forb_speed_over_50', 'forb_speed_over_80', 'info_crosswalk', 'prio_give_way', 'warn_other_dangers', 'prio_stop']

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            with torch.no_grad():
                model_out,_,_ = self.forward(torch.zeros(1, 3, s, s))
                detects= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        image = x
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            if i == self.seg_da_idx:
                x = block(x[0])
            elif i == self.seg_ll_idx:
                x = block(x[1])
            else:
                x = block(x)

            if i == self.seg_da_idx or i == self.seg_ll_idx:
                out.append(x)
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out
            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs): 
    model_cfg = sc_ch_dict[cfg.config]
    m_block_cfg = TriLiteNet(model_cfg)
 
    model = MultiTaskModel(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from thop import profile
    # from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 384, 640))
    gt_ = torch.rand((1, 2, 384, 640))
    metric = SegmentationMetric(2)
    with torch.no_grad():
        model_out,Da_fmap, LL_fmap = model(input_)

 
