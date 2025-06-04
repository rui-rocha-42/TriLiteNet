import torch
from lib.utils import is_parallel
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
from sklearn.cluster import DBSCAN
import torch.nn.functional as F

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def find_3_positive(p, targets, det, cfg):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    indices, anch = [], []
    gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < cfg.TRAIN.ANCHOR_THRESHOLD  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        anch.append(anchors[a])  # anchors

    return indices, anch
def build_targets_ota(cfg, predictions, targets, model, imgs):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[model.module.detector_index] if is_parallel(model) \
        else model.model[model.detector_index]  # Detect() module
    device = torch.device(targets.device)
    indices, anch = find_3_positive(predictions, targets, det, cfg)
    matching_bs = [[] for pp in predictions]
    matching_as = [[] for pp in predictions]
    matching_gjs = [[] for pp in predictions]
    matching_gis = [[] for pp in predictions]
    matching_targets = [[] for pp in predictions]
    matching_anchs = [[] for pp in predictions]
    
    nl = len(predictions)  


    for batch_idx in range(predictions[0].shape[0]):
        
        b_idx = targets[:, 0]==batch_idx
        this_target = targets[b_idx]
        if this_target.shape[0] == 0:
            continue
            
        txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
        txyxy = xywh2xyxy(txywh)

        pxyxys = []
        p_cls = []
        p_obj = []
        from_which_layer = []
        all_b = []
        all_a = []
        all_gj = []
        all_gi = []
        all_anch = []
        
        for i, pi in enumerate(predictions):
            
            b, a, gj, gi = indices[i]
            idx = (b == batch_idx)
            b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
            all_b.append(b)
            all_a.append(a)
            all_gj.append(gj)
            all_gi.append(gi)
            all_anch.append(anch[i][idx])
            from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
            
            fg_pred = pi[b, a, gj, gi]                
            p_obj.append(fg_pred[:, 4:5])
            p_cls.append(fg_pred[:, 5:])
            
            grid = torch.stack([gi, gj], dim=1)
            pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * det.stride[i] #/ 8.
            pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * det.stride[i] #/ 8.
            pxywh = torch.cat([pxy, pwh], dim=-1)
            pxyxy = xywh2xyxy(pxywh)
            pxyxys.append(pxyxy)
        
        pxyxys = torch.cat(pxyxys, dim=0)
        if pxyxys.shape[0] == 0:
            continue
        p_obj = torch.cat(p_obj, dim=0)
        p_cls = torch.cat(p_cls, dim=0)
        from_which_layer = torch.cat(from_which_layer, dim=0)
        all_b = torch.cat(all_b, dim=0)
        all_a = torch.cat(all_a, dim=0)
        all_gj = torch.cat(all_gj, dim=0)
        all_gi = torch.cat(all_gi, dim=0)
        all_anch = torch.cat(all_anch, dim=0)
    
        pair_wise_iou = box_iou(txyxy, pxyxys)

        pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

        top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
        dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

        gt_cls_per_image = (
            F.one_hot(this_target[:, 1].to(torch.int64), det.nc)
            .float()
            .unsqueeze(1)
            .repeat(1, pxyxys.shape[0], 1)
        )

        num_gt = this_target.shape[0]
        cls_preds_ = (
            p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        )

        y = cls_preds_.sqrt_()
        pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
            torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
        ).sum(-1)
        del cls_preds_
    
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_iou_loss
        )

        matching_matrix = torch.zeros_like(cost, device=device)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del top_k, dynamic_ks
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    
        from_which_layer = from_which_layer[fg_mask_inboxes]
        all_b = all_b[fg_mask_inboxes]
        all_a = all_a[fg_mask_inboxes]
        all_gj = all_gj[fg_mask_inboxes]
        all_gi = all_gi[fg_mask_inboxes]
        all_anch = all_anch[fg_mask_inboxes]
    
        this_target = this_target[matched_gt_inds]
    
        for i in range(nl):
            layer_idx = from_which_layer == i
            matching_bs[i].append(all_b[layer_idx])
            matching_as[i].append(all_a[layer_idx])
            matching_gjs[i].append(all_gj[layer_idx])
            matching_gis[i].append(all_gi[layer_idx])
            matching_targets[i].append(this_target[layer_idx])
            matching_anchs[i].append(all_anch[layer_idx])

    for i in range(nl):
        if matching_targets[i] != []:
            matching_bs[i] = torch.cat(matching_bs[i], dim=0)
            matching_as[i] = torch.cat(matching_as[i], dim=0)
            matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
            matching_gis[i] = torch.cat(matching_gis[i], dim=0)
            matching_targets[i] = torch.cat(matching_targets[i], dim=0)
            matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
        else:
            matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

    return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

def build_targets(cfg, predictions, targets, model):
    '''
    predictions
    [16, 3, 32, 32, 85]
    [16, 3, 16, 16, 85]
    [16, 3, 8, 8, 85]
    torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
    [32,32,32,32]
    [16,16,16,16]
    [8,8,8,8]
    targets[3,x,7]
    t [index, class, x, y, w, h, head_index]
    '''
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[model.module.detector_index] if is_parallel(model) \
        else model.model[model.detector_index]  # Detect() module
    # print(type(model))
    # det = model.model[model.detector_index]
    # print(type(det))
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    
    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    
    for i in range(det.nl):
        anchors = det.anchors[i] #[3,2]
        gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        # Match targets to anchors
        t = targets * gain

        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < cfg.TRAIN.ANCHOR_THRESHOLD  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp(0, gain[3] - 1).long(), gi.clamp(0, gain[2] - 1).long()))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

def morphological_process(image, kernel_size=5, func_type=cv2.MORPH_CLOSE):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, func_type, kernel, iterations=1)

    return closing

def connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # print(gray_image.dtype)
    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)

def if_y(samples_x):
    for sample_x in samples_x:
        if len(sample_x):
            # if len(sample_x) != (sample_x[-1] - sample_x[0] + 1) or sample_x[-1] == sample_x[0]:
            if sample_x[-1] == sample_x[0]:
                return False
    return True
    
def fitlane(mask, sel_labels, labels, stats):
    H, W = mask.shape
    for label_group in sel_labels:
        states = [stats[k] for k in label_group]
        x, y, w, h, _ = states[0]
        # if len(label_group) > 1:
        #     print('in')
        #     for m in range(len(label_group)-1):
        #         labels[labels == label_group[m+1]] = label_group[0]
        t = label_group[0]
        # samples_y = np.linspace(y, H-1, 30)
        # else:
        samples_y = np.linspace(y, y+h-1, 30)
        
        samples_x = [np.where(labels[int(sample_y)]==t)[0] for sample_y in samples_y]

        if if_y(samples_x):
            samples_x = [int(np.mean(sample_x)) if len(sample_x) else -1 for sample_x in samples_x]
            samples_x = np.array(samples_x)
            samples_y = np.array(samples_y)
            samples_y = samples_y[samples_x != -1]
            samples_x = samples_x[samples_x != -1]
            func = np.polyfit(samples_y, samples_x, 2)
            x_limits = np.polyval(func, H-1)
            # if (y_max + h - 1) >= 720:
            if x_limits < 0 or x_limits > W:
            # if (y_max + h - 1) > 720:
                # draw_y = np.linspace(y, 720-1, 720-y)
                draw_y = np.linspace(y, y+h-1, h)
            else:
                # draw_y = np.linspace(y, y+h-1, y+h-y)
                draw_y = np.linspace(y, H-1, H-y)
            draw_x = np.polyval(func, draw_y)
            # draw_y = draw_y[draw_x < W]
            # draw_x = draw_x[draw_x < W]
            draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
            cv2.polylines(mask, [draw_points], False, 1, thickness=15)
        else:
            # if ( + w - 1) >= 1280:
            samples_x = np.linspace(x, W-1, 30)
            # else:
            #     samples_x = np.linspace(x, x_max+w-1, 30)
            samples_y = [np.where(labels[:, int(sample_x)]==t)[0] for sample_x in samples_x]
            samples_y = [int(np.mean(sample_y)) if len(sample_y) else -1 for sample_y in samples_y]
            samples_x = np.array(samples_x)
            samples_y = np.array(samples_y)
            samples_x = samples_x[samples_y != -1]
            samples_y = samples_y[samples_y != -1]
            try:
                func = np.polyfit(samples_x, samples_y, 2)
            except:
                pass
            # y_limits = np.polyval(func, 0)
            # if y_limits > 720 or y_limits < 0:
            # if (x + w - 1) >= 1280:
            #     draw_x = np.linspace(x, 1280-1, 1280-x)
            # else:
            y_limits = np.polyval(func, 0)
            if y_limits >= H or y_limits < 0:
                draw_x = np.linspace(x, x+w-1, w+x-x)
            else:
                y_limits = np.polyval(func, W-1)
                if y_limits >= H or y_limits < 0:
                    draw_x = np.linspace(x, x+w-1, w+x-x)
                # if x+w-1 < 640:
                #     draw_x = np.linspace(0, x+w-1, w+x-x)
                else:
                    draw_x = np.linspace(x, W-1, W-x)
            draw_y = np.polyval(func, draw_x)
            draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
            cv2.polylines(mask, [draw_points], False, 1, thickness=15)
    return mask

def connect_lane(image, shadow_height=0):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    if shadow_height:
        image[:shadow_height] = 0
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)
    # ratios = []
    selected_label = []
    
    for t in range(1, num_labels, 1):
        _, _, _, _, area = stats[t]
        if area > 400:
            selected_label.append(t)
    if len(selected_label) == 0:
        return mask
    else:
        split_labels = [[label,] for label in selected_label]
        mask_post = fitlane(mask, split_labels, labels, stats)
        return mask_post








