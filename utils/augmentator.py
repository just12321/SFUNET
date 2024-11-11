import cv2
import numpy as np
from collections.abc import Sequence
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc = mean, scale = sd)

class AlignedAugmentator:
    def __init__(self, ratio = [0.3, 1.], target_size = (256, 256), flip = True,
                 distruction = 'Uniform', gs_center = 0.8, gs_sd = 0.4,
                 color_augmentator = None):
        self.ratio = ratio
        self.target_size = target_size
        self.flip = flip
        self.distruction = distruction
        self.gaussian = get_truncated_normal(mean = gs_center, sd = gs_sd, low = ratio[0], upp = ratio[1])
        self.color_augmentator = color_augmentator
        self.pad = PadIfNeeded()
    
    def __call__(self, image, mask):
        if self.distruction == 'Uniform':
            hr, wr = np.random.uniform(*self.ratio), np.random.uniform(*self.ratio)
        elif self.distruction == 'Gaussian':
            hr, wr = self.gaussian.rvs(2)
        H, W = image.shape[:2]
        h, w = int(hr * H), int(wr * W)
        if hr > 1 or wr > 1:
            image, mask = self.pad(image, mask, hr, wr)
            H, W = image.shape[:2]
        y1 = np.random.randint(0, H - h)
        x1 = np.random.randint(0, W - w)
        y2 = y1 + h
        x2 = x1 + w

        image_crop = image[y1:y2, x1:x2, :]
        image_crop = cv2.resize(image_crop, self.target_size)
        mask_crop = mask[y1:y2, x1:x2, :].astype(np.uint8)
        mask_crop = (cv2.resize(mask_crop, self.target_size)).astype(np.int32)
        
        if self.flip:
            if np.random.rand() < 0.3:
                image_crop = np.flip(image_crop, 0)
                mask_crop = np.flip(mask_crop, 0)
            if np.random.rand() < 0.3:
                image_crop = np.flip(image_crop, 1)
                mask_crop = np.flip(mask_crop, 1)
        
        image_crop = np.ascontiguousarray(image_crop)
        mask_crop = np.ascontiguousarray(mask_crop)

        if self.color_augmentator is not None:
            image_crop = self.color_augmentator(image = image_crop)['image']
        
        return {'image': image_crop, 'mask': mask_crop}
    


class UniformRandomResize:
    def __init__(self, scale_range = (0.75, 1.25)):
        if not isinstance(scale_range, Sequence) or len(scale_range) != 2:
            raise ValueError("It should have 2 values")
        self.scale_range = scale_range
    
    def __call__(self, image, mask):
        H, W = image.shape[:2]
        scale = np.random.uniform(*self.scale_range)
        h, w = int(scale * H), int(scale * W)
        image = cv2.resize(image, (w, h))
        mask = cv2.resize(mask, (w, h))
        return image, mask

class PadIfNeeded:
    def __init__(self, min_height = 256, min_width = 256):
        self.min_height = min_height
        self.min_width = min_width

    def __call__(self, image, mask, h_r = None, w_r = None):
        H, W = image.shape[:2]
        hr = self.min_height / H if h_r is None else h_r
        wr = self.min_width / W if w_r is None else w_r
        if hr > 1:
            new_h = int(H * hr) + 1 if h_r else self.min_height
            pad_h = new_h - H
            pad_h1 = np.random.randint(0, pad_h)
            pad_h2 = pad_h - pad_h1
            image = np.pad(image, ((pad_h1, pad_h2), (0, 0), (0, 0)), 'constant', constant_values=1)
            mask = np.pad(mask, ((pad_h1, pad_h2), (0, 0), (0, 0)), 'constant')
        if wr > 1:
            new_w = int(W * wr) + 1 if w_r else self.min_width
            pad_w = new_w - W
            pad_w1 = np.random.randint(0, pad_w)
            pad_w2 = pad_w - pad_w1
            image = np.pad(image, ((0, 0), (pad_w1, pad_w2), (0, 0)), 'constant', constant_values=1)
            mask = np.pad(mask, ((0, 0), (pad_w1, pad_w2), (0, 0)), 'constant')
        return image, mask
