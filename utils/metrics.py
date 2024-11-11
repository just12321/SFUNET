import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, _ni_support, generate_binary_structure, distance_transform_edt

def Accuracy(output, target):
    """
    Binary classification accuracy: Compute accuracy per image in the batch and average them.
    """
    pred = torch.round(output).int()
    target = torch.round(target).int()
    correct_per_image = (pred == target).sum(dim=(1, 2, 3)).float()
    accuracy_per_image = correct_per_image / target[0].numel()

    return accuracy_per_image.mean().item()

def Iou(output, target):
    """
    Binary classification IoU: Compute IoU per image in the batch and average them.
    """
    pred = torch.round(output).int()
    target = torch.round(target).int()

    intersect = (pred & target).sum(dim=(1, 2, 3)).float()
    union = (pred | target).sum(dim=(1, 2, 3)).float()

    iou_per_image = intersect / (union + 1e-6)

    return iou_per_image.mean().item()

def mIoU(output, target):
    """
    Mean IoU for binary classification
    """
    iou_fg = Iou(output, target)
    iou_bg = Iou(1 - output, 1 - target)
    
    mean_iou = (iou_fg + iou_bg) / 2
    return mean_iou

def BoundaryIou(output, target, radius=3):
    """
    Binary classify boundary IoU
    """
    pred = torch.round(output).int()
    pred_boundary = pred ^ torch.tensor(binary_erosion(pred, torch.ones(1, 1, radius, radius)), device=target.device)
    target_binary = torch.round(target)
    target_boundary = target_binary ^ torch.tensor(binary_erosion(target_binary, torch.ones(1, 1, radius, radius)), device=target.device)
    return Iou(pred_boundary, target_boundary)

def ErodeIou(output, target, radius=3):
    """
    Binary classify erode IoU
    """
    pred = torch.round(output).int()
    pred_erode = torch.tensor(binary_erosion(pred, torch.ones(1, 1, radius, radius)), device=target.device)
    target_binary = torch.round(target).int()
    target_erode = torch.tensor(binary_erosion(target_binary, torch.ones(1, 1, radius, radius)), device=target.device)
    return Iou(pred_erode, target_erode)

def FScore(output, target, beta=1):
    """
    Binary classification F-score: Compute F-score per image in the batch and average them.
    """
    pred = torch.round(output).int()
    target = torch.round(target).int()

    intersect = (pred & target).sum(dim=(1, 2, 3)).float()
    pred_sum = pred.sum(dim=(1, 2, 3)).float()
    target_sum = target.sum(dim=(1, 2, 3)).float()

    precision = intersect / (pred_sum + 1e-6)
    recall = intersect / (target_sum + 1e-6)

    f_score_per_image = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-6)

    return f_score_per_image.mean().item()


def Dice(output, target):
    """
    Binary classify Dice
    """
    return FScore(output, target)

def Recall(output, target):
    """
    Binary classification Recall: Compute recall per image in the batch and average them.
    """
    pred = torch.round(output).int()
    target = torch.round(target).int()

    intersect = (pred & target).sum(dim=(1, 2, 3)).float()
    target_sum = target.sum(dim=(1, 2, 3)).float()

    recall_per_image = intersect / (target_sum + 1e-6)
    return recall_per_image.mean().item()

def Precision(output, target):
    """
    Binary classification Precision: Compute precision per image in the batch and average them.
    """
    pred = torch.round(output).int()
    target = torch.round(target).int()

    intersect = (pred & target).sum(dim=(1, 2, 3)).float()
    pred_sum = pred.sum(dim=(1, 2, 3)).float()

    precision_per_image = intersect / (pred_sum + 1e-6)
    return precision_per_image.mean().item()

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError(
            "The first supplied array does not contain any binary object."
        )
    if 0 == np.count_nonzero(reference):
        raise RuntimeError(
            "The second supplied array does not contain any binary object."
        )

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds

def robust_hd(result, reference, voxelspacing=None, connectivity=1, percent=100):
    """
    https://github.com/loli/medpy/blob/master/medpy/metric/binary.py#L371 hd95
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd = np.percentile(np.hstack((hd1, hd2)), percent)
    return hd

def HD95(output, target):
    output = np.array(output.cpu())
    target = np.array(target.cpu())
    return robust_hd(output, target, percent=95)