from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def get_dims_with_exclusion(dim:int, exclusion = None):
    dims = list(range(dim))
    if exclusion != None:
        dims.remove(exclusion)
    return dims


def _iou(pred, target, size_average=True):
    # Compute intersection and union
    intersection = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection
    
    # Compute IoU for each sample in the batch
    iou = intersection / union
    
    # Compute the IoU loss (1 - IoU) for each sample
    iou_loss = 1 - iou
    
    # Average the IoU loss over the batch
    if size_average:
        return iou_loss.mean()
    else:
        return iou_loss.sum()
 
class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average
 
    def forward(self, pred, target):
 
        return _iou(pred, target, self.size_average) #size_average即对结果求平均值。
 
def IOU_loss(pred,label,from_logits=False, reduction='mean'):
    if not from_logits:
        pred = torch.sigmoid(pred)
    iou_loss = IOU(size_average=reduction=='mean')
    iou_out = iou_loss(pred, label)
    return iou_out

def __dice_loss(input: torch.FloatTensor, target: torch.LongTensor, weights: torch.FloatTensor = None, k: int = 0, eps: float = 0.0001):
    """
    Returns the Generalized Dice Loss Coefficient associated to the input and target tensors, as well as to the input weights,\
    in case they are specified.

    Args:
        input (torch.FloatTensor): CHW tensor containing the classes predicted for each pixel.
        target (torch.LongTensor): CHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        weights (torch.FloatTensor): 2D tensor of size C, containing the weight of each class, if specified.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    """  
    n_classes = input.size()[0]

    if weights is not None:
        for c in range(n_classes):
            intersection = (input[c] * target[c] * weights[c]).sum()
            union = (weights[c] * (input[c] + target[c])).sum() + eps
    else:
        intersection = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps    

    gd = (2 * intersection.float() + eps) / union.float()
    return 1 - (gd / (1 + k*(1-gd)))

def dice_loss(input: torch.FloatTensor, target: torch.LongTensor, from_logits=False, use_weights: bool = False, k: int = 0, eps: float = 0.0001, reduction='mean'):
    """
    Returns the Generalized Dice Loss Coefficient of a batch associated to the input and target tensors. In case `use_weights` \
        is specified and is `True`, then the computation of the loss takes the class weights into account.

    Args:
        input (torch.FloatTensor): NCHW tensor containing the probabilities predicted for each class.
        target (torch.LongTensor): NCHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        use_weights (bool): specifies whether to use class weights in the computation or not.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    """
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # Multiple class case
    n_classes = input.size()[1]
    if n_classes != 1:
        # Convert target to one hot encoding
        target = F.one_hot(target, n_classes).squeeze()
        if target.ndim == 3:
            target = target.unsqueeze(0)
        target = torch.transpose(torch.transpose(target, 2, 3), 1, 2).type(torch.FloatTensor).cuda().contiguous()
        input = torch.softmax(input, dim=1)
    else:
        if not from_logits:input = torch.sigmoid(input)   

    class_weights = None
    for i, c in enumerate(zip(input, target)):
        if use_weights:
            class_weights = torch.pow(torch.sum(c[1], (1,2)) + eps, -2)
        s = s + __dice_loss(c[0], c[1], class_weights, k=k)

    res = s / (i + 1)
    if reduction == 'mean':
        return res.mean()
    if reduction == 'none':
        return res

class DiceLoss(nn.Module):
    def __init__(self, from_logits=False, use_weights=False, k=0, eps=0.0001):
        super().__init__()
        self._from_logits = from_logits
        self._use_weights = use_weights
        self._k = k
        self._eps = eps
        
    def forward(self, pred:torch.Tensor, label:torch.Tensor):
        return dice_loss(pred, label, self._from_logits, self._use_weights, self._k, self._eps)
    
class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1., with_logits = False,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.with_logits = with_logits
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.with_logits:
            x = torch.sigmoid(x)
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        # if self.ddp and self.batch_dice:
        #     intersect = AllGatherGrad.apply(intersect).sum(0)
        #     sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
        #     sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc

def focal_loss(pred:torch.Tensor, label:torch.Tensor, alpha = 0.25, gamma = 2., from_logits = False, batch_axis = 0, weight = 1,
                eps = 1e-9, size_average = True, scale = 1., ignore_label = -1, reduction='mean'):
    size_average = size_average and reduction != 'none'
    sample_weight = label != ignore_label
    if not from_logits:
        pred = torch.sigmoid(pred)
    alpha = torch.where(label > 0.5, alpha * sample_weight, (1 - alpha) * sample_weight)
    pt = torch.where(sample_weight, torch.abs(label - pred), torch.zeros_like(pred))
    beta = pt ** gamma
    loss = -alpha * beta * torch.log(torch.min(1 - pt + eps, torch.ones(1, dtype = torch.float).to(pt.device))) * weight
    if size_average:
        tsum = torch.sum(sample_weight, dim = get_dims_with_exclusion(sample_weight.dim(), batch_axis))
        loss = torch.sum(loss, dim = get_dims_with_exclusion(loss.dim(), batch_axis)) / (tsum + eps)
        return (scale * loss).mean()
    else:
        return scale * loss

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2., from_logits = False, batch_axis = 0, weight = None,
                 eps = 1e-9, size_average = True, scale = 1., ignore_label = -1):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight else 1.
        self._batch_axis = batch_axis
        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
    
    def forward(self, pred:torch.Tensor, label:torch.Tensor):
        return focal_loss(pred, label, self._alpha, self._gamma, self._from_logits, self._batch_axis, self._weight,
                          self._eps, self._size_average, self._scale, self._ignore_label
        )
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])  #双星号：幂的意思。 双//：表示向下取整，有一方是float型时，结果为float。  exp()返回e的x次方。
    return gauss/gauss.sum()
 
 
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) #unsqueeze（x）增加维度x
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  #t() 将tensor进行转置。  x.mm(self.y) 将x与y相乘。
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
#返回的均值。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    #求像素的动态范围
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    #求img1，img2的均值。
    padd = 0
    (_, channel, height, width) = img1.size() # _ 为批次batch大小。
        #定义卷积核window
    if window is None:
        real_size = min(window_size, height, width) #求最小值，是为了保证卷积核尺寸和img1，img2尺寸相同。
        window = create_window(real_size, channel=channel).to(img1.device)
 
        #空洞卷积：有groups代表是空洞卷积；  F.conv2d(输入图像tensor，卷积核tensor, ...)是卷积操作。
        #mu1为img1的均值；mu2为img2的均值。
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel) #groups控制分组卷积，默认不分组，即为1.  delition默认为1.
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel) #conv2d输出的是一个tensor-新的feature map。
 
        #mu1_sq:img1均值的平方。 mu2_sq:img2均值的平方
    mu1_sq = mu1.pow(2) #对mu1中的元素逐个2次幂计算。
    mu2_sq = mu2.pow(2)
        #img1,img2均值的乘积。
    mu1_mu2 = mu1 * mu2
 
    #x的方差σx²
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    #y的方差σy²
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    #求x,y的协方差σxy
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    #维持稳定的两个变量
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    #v1:2σxy+C2
    v1 = 2.0 * sigma12 + C2
    #v2:σx²+σy²+C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity   #对比敏感度
 
    #ssim_map为img1,img2的相似性指数。
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    #求平均相似性指数。 ??
    if size_average: #要求平均时
        ret = ssim_map.mean()
    else: #不要求平均时
        ret = ssim_map.mean(1).mean(1).mean(1) #mean(1) 求维度1的平均值
 
    if full:
        return ret, cs
    return ret

def softIou(pred:torch.Tensor, label:torch.Tensor, from_sigmoid = False, ignore_label = -1):
    if from_sigmoid:
        pred = pred.sigmoid()
    label = label.view(pred.size())
    sample_weight = label != ignore_label
    loss = 1. - torch.einsum('bhw->', pred * label * sample_weight) / (torch.einsum('bhw->', torch.max(pred, label) * sample_weight) + 1e-8)
    return loss

class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid = False, ignore_label = -1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
    
    def forward(self, pred:torch.Tensor, label:torch.Tensor):
        return softIou(pred, label, self._from_sigmoid, self._ignore_label)
    
def contain_loss(pred:torch.Tensor, label:torch.Tensor, with_logits=True):
    if with_logits:
        target = pred[label == 1]
        loss = F.binary_cross_entropy(target, torch.ones_like(target))
        return loss
    loss = torch.relu(pred) - label * pred + label * F.softplus(-pred)
    return loss.mean()

class ContainLoss(nn.Module):
    def __init__(self, with_logits=True):
        super().__init__()
        self.with_logits = with_logits

    def forward(self, pred:torch.Tensor, label:torch.Tensor):
        return contain_loss(pred, label, self.with_logits)

def pixel_penalty(pred:torch.Tensor, img:torch.Tensor, with_logits=True):
    if not with_logits:  
        pred = pred.sigmoid()
    tar_mean = (pred*img).mean((1,2,3), keepdim=True)
    other_mean = ((1-pred)*img).mean((1,2,3), keepdim=True)
    return -F.l1_loss(tar_mean, other_mean)    

def keep_loss(pred:torch.Tensor, label:torch.Tensor, img:torch.Tensor=None, pos_weight:list[float]=[1.]):
    # kernels = torch.zeros((4, 3, 3), device=pred.device)
    # kernels[0, 0, 1] = -1
    # kernels[0, 1, 1] = 1
    # kernels[1, 1, 0] = -1
    # kernels[1, 1, 1] = 1
    # kernels[2, 0, 1] = -1
    # kernels[2, 1, 1] = 2
    # kernels[2, 2, 1] = -1
    # kernels[3, 1, 0] = -1
    # kernels[3, 1, 1] = 2
    # kernels[3, 1, 2] = -1

    # kernels = torch.zeros((4, 2, 2), device=pred.device)
    # kernels[0, 0, 1] = -1
    # kernels[0, 1, 1] = 1
    # kernels[1, 1, 0] = -1
    # kernels[1, 1, 1] = 1
    # kernels[2, 0, 0] = -1
    # kernels[2, 1, 1] = 1
    # kernels[3, 0, 1] = -1
    # kernels[3, 1, 0] = 1
    # pred_ = torch.relu(pred * (label - 1))
    # tmp_grad = torch.abs(F.conv2d(pred_, kernels.unsqueeze(1), padding='same')).sum((0,1))
    # img_grad = (torch.abs(F.conv2d(img.mean(1, keepdim=True), kernels.unsqueeze(1), padding='same')) * (pred < 0)).sum((0,1))
    # # loss_connect = tmp_grad[0].mean() + tmp_grad[1].mean() - tmp_grad[2].mean() - tmp_grad[3].mean() #- (tmp_grad[0]**2 + tmp_grad[1]**2).mean() #
    # loss_connect = tmp_grad.mean()
    # loss_edge = img_grad.mean()
    tmp = pred[label > 0]
    loss_contain = F.binary_cross_entropy_with_logits(tmp, torch.ones_like(tmp), pos_weight=torch.tensor(pos_weight, device=tmp.device)) + pred.sigmoid().mean()
    # penalty = pixel_penalty(pred, img, with_logits=False)
    return loss_contain
    # return loss_connect + loss_edge + loss_contain 

class KeepLoss(nn.Module):
    def __init__(self, with_logits=True):
        super().__init__()
        self.with_logits = with_logits

    def forward(self, pred:torch.Tensor, label:torch.Tensor):
        return keep_loss(pred, label, self.with_logits)
    
def assbce_loss(pred:torch.Tensor, label:torch.Tensor, with_logits=False):
    if not with_logits:
        pred = torch.sigmoid(pred)
    tmp = pred[label > 0]
    mean = tmp.mean()
    std = tmp.var()
    thre = mean + std
    idx = torch.logical_or(pred > thre, pred < 1 - thre)
    return F.binary_cross_entropy(pred[idx], label[idx])

class ASSBCELoss(nn.Module):
    """
    Refer to 《Adaptive Sample Selection for Robust Learning Under Label Noise》
    see: https://openaccess.thecvf.com/content/WACV2023/papers/Patel_Adaptive_Sample_Selection_for_Robust_Learning_Under_Label_Noise_WACV_2023_paper.pdf
    """
    def __init__(self, with_logits=False):
        super().__init__()
        self.with_logits = with_logits

    def forward(self, pred:torch.Tensor, label:torch.Tensor):
        return assbce_loss(pred, label, self.with_logits)

def gce_loss(pred: torch.Tensor, target: torch.Tensor, q: float = 0.7, with_logits=False):
    """
    Compute the Generalized Cross-Entropy (GCE) loss.
    
    Parameters:
    pred (torch.Tensor): The predicted values (logits).
    target (torch.Tensor): The ground truth labels.
    q (float): The parameter controlling the robustness (default: 0.7).
    
    Returns:
    torch.Tensor: The GCE loss.
    """
    pred_prob = torch.sigmoid(pred) if not with_logits else pred
    loss = (1 - (target * pred_prob + (1 - target) * (1 - pred_prob)) ** q) / q
    return loss.mean()

class DiffLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred:torch.Tensor, label:torch.Tensor, inc:torch.Tensor):
        return diff_loss(pred, label, inc)

def diff_loss(pred: torch.Tensor, label: torch.Tensor, inc: torch.Tensor = None, alpha=2., gamma=0.5):
    sigma_pred = torch.sigmoid(pred)
    with torch.no_grad():
        nf = torch.where((label == 1) & (pred < 0), 1 - sigma_pred, 0.5)
        nt = torch.where((label == 0) & (pred > 0), sigma_pred, 0.5)
        pf = sigma_pred[(pred < 0) & (label < 1)]
        pt = sigma_pred[(pred >= 0) & (label > 0)]
        lambda_pf = (pf.mean() - pf.std() / 2) if pf.numel() > 100 else 0
        lambda_pt = (pt.mean() + pt.std() / 2) if pt.numel() > label.sum() * 0.5 else 1

        weight = alpha * ((lambda_pf <= sigma_pred).float() * (sigma_pred <= lambda_pt).float() * nt * nf) ** gamma
    
    bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    bce = (bce * weight).mean()
    
    return bce

# def diff_loss(pred:torch.Tensor, label:torch.Tensor, inc:torch.Tensor=None, alpha=2., gamma=0.5):
#     sigma_pred = torch.sigmoid(pred)
#     with torch.no_grad():
#         nf = torch.where((label == 1) & (pred < 0), 1 - sigma_pred, 0.5)
#         nt = torch.where((label == 0) & (pred > 0), sigma_pred, 0.5)
#         pf = sigma_pred[(pred < 0) & (label < 1)]
#         pt = sigma_pred[(pred >= 0) & (label > 0)]
#         lambda_pf = (pf.mean() + pf.std()) if pf.numel() > 100 else 0
#         lambda_pt = (pt.mean() - pt.std()) if pt.numel() > label.sum() * 0.5 else 1
#         # nf = F.avg_pool2d(nf, 3, 1, 1)
#         # nt = F.avg_pool2d(nt, 3, 1, 1)
#         # nf = nf * (nf <= 1 - lambda_pf).float()
#         # nt = nt * (nt <= lambda_pt).float()

#         # if Iou(sigma_pred, label) > 0.5:
#         # binary_pred = torch.where((pred >= 0) | (label > 0), 1., 0.)
#         # kernel = torch.tensor([[-1, -1, -1],
#         #                     [-1,  8, -1],
#         #                     [-1, -1, -1]], device=pred.device).float().unsqueeze(0).unsqueeze(0)
#         # edges = F.conv2d(binary_pred, kernel, padding=1)
#         # edges = (edges.abs() < 5).float()
#         # else:
#         #     edges = 1

#         weight = alpha * ((lambda_pf <= sigma_pred).float() * (sigma_pred <= lambda_pt).float() * nt * nf ) ** gamma
#     bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
#     bce = (bce * weight.clone().detach()).mean()
#     if inc is None: 
#         return bce
#     add = pred[(label < 1) & (inc >= 0)]
#     support = (-add).relu().mean()
#     return support + bce
#     add = pred.relu().sigmoid() * (1 - label)
#     support = - (add * (inc > 0).float() * (pred > 0).float()).mean()
    
#     com = keep_loss(inc, label, None)
#     penalty = support + com
    
#     return penalty + bce

def simple_loss(pred:torch.Tensor, label:torch.Tensor, step=3):
    eroded = label.float()
    for _ in range(step):
        eroded = -F.max_pool2d(-label, 3, stride=1, padding=1)
    edge_tensor = label - eroded
    return keep_loss(pred, edge_tensor, pos_weight=[1.1])

def lead_loss(pred:torch.Tensor, label:torch.Tensor, inc:torch.Tensor=None):
    keep = pred[inc > 0]
    dilated = label.float()
    for _ in range(3):
        dilated = F.max_pool2d(dilated, 3, stride=1, padding=1)
    edge_tensor = dilated - label
    drop = pred[edge_tensor > 0]
    return F.binary_cross_entropy_with_logits(keep, torch.ones_like(keep)) + F.binary_cross_entropy_with_logits(drop, torch.zeros_like(drop))