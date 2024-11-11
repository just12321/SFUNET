import functools
import inspect
import numpy as np
import os
from pathlib import Path, WindowsPath
import shutil
from typing import Dict, Optional, cast
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
import einops as eop
from matplotlib import colormaps as cms, pyplot as plt
from matplotlib import cm

from utils.general import colorstr
from utils.log import add_logging, logger as Logger

def get_next_exp(path:WindowsPath, modelname:str, expname:str):
    """
    Return the name of the next experiment of certain model, e.g., there are 'Unet000', 'Unet001' in the fold, then it should return path / 'Unet002'
    """
    p = path / 'info' / modelname
    if not p.exists():
        p.mkdir(parents=True)
    files = [f for f in p.iterdir() if expname == f.stem[:-3]]
    if not files:
        idx = 0
    else:
        idx = max([int(f.stem.split(expname)[1]) for f in files]) + 1
    postfix = f'{expname}{idx:0>3d}'
    log_path = p / postfix / 'log'
    vision_path = path / 'vision' / modelname / postfix
    checkpoint_path = p / postfix / 'checkpoint'
    log_path.mkdir(exist_ok=True, parents=True)
    vision_path.mkdir(exist_ok=True, parents=True)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    return {
        'log': log_path,
        'vision': vision_path,
        'checkpoint': checkpoint_path
    }

def mask_to_rgb(mask, colormap=None):
    colors = {
        0: [0, 0, 0],  
        1: [255, 255, 255]  
    } if colormap is None else colormap
    height, width, _ = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for ch in range(mask.shape[-1]):
        for cid, color in colors.items():
            mask_indices = np.where(mask[..., ch] == cid)
            rgb_image[mask_indices[0], mask_indices[1]] = color
    return rgb_image

def save_as_rgb(src, dest):
    mask = np.array(Image.open(src))
    rgb_image = mask_to_rgb(mask)
    Image.fromarray(rgb_image).save(dest)

def filter(mask, threshold=100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > threshold:
            filtered_mask[labels == label] = 1
    return filtered_mask

def binary(mask, threshold=0.5):
    out = torch.where(mask >= threshold, torch.tensor(1.0), torch.tensor(0.0))
    return out

def smooth(mask, step=3):
    kernel = torch.tensor([[0.0, 1.0, 0.0],
                           [1.0, 1.0, 1.0],
                           [0.0, 1.0, 0.0]], device=mask.device).unsqueeze(0).unsqueeze(0) 
    for _ in range(step):
        mask = torch.clamp(F.conv2d(mask, kernel, padding=1), 0, 1)
    for _ in range(step):
        mask = F.relu(F.conv2d(mask, kernel, padding=1) - 4)
    return mask

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_config():
    return edict({
        'weights': None,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'cpt_period': 200,
        'vision_period': 10,
        'batch_size': 4,
        'num_epoch': 2000,
        'seed': None,
        'expRoot': Path('./exp'),
        'resume': False,
        'expName': 'Model'
    })

def convert_value(value):
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        # Check if the callable is a lambda function
        if inspect.isfunction(value) and value.__name__ == "<lambda>":
            raise ValueError("Lambda functions are not allowed.")
        return value.__name__
    return value

def apply_config(config):
    if config.seed is not None:
        setup_seed(config.seed)
    expPath = get_next_exp(config.expRoot, config.modelName, config.expName)
    config.logPath = expPath['log']
    config.visionPath = expPath['vision']
    config.checkpointPath = expPath['checkpoint']
    add_logging(config.logPath, config.expName)

    shutil.copy2(config.mainFile, expPath['log'] / 'main.py')
    shutil.copy2(config.modelFile, expPath['log'] / f'{config.modelName}.py')
    with open(expPath['log'] / 'model.py', 'w') as f:
        f.write(f"modelName='{config.modelName}'\n\n")
        f.write("args={\n")
        for k, v in config.args.items():
            try:
                tv = convert_value(v)
                f.write(f"\t'{k}': {tv},\n")
            except:
                # might be an instance
                pass
        f.write("}")

def print_config(config):
    text = f'{colorstr("Configurations:")}\n\t'+'\n\t'.join('\n'.join(([f'{k}: {v}' for k, v in config.items()])).split('\n'))
    Logger.info(text)
    return text

def overlay_mask(img: torch.Tensor, mask: torch.Tensor, colormap: str = "jet", alpha: float = 0.7) -> torch.Tensor:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cms.get_cmap(colormap)
    overlay = (cmap(mask ** 2)[:, :, :3])
    overlayed_img = (alpha * img + (1 - alpha) * overlay)
    return overlayed_img

def grad_cam(feature, grads):
    grads = F.adaptive_avg_pool2d(grads, 1)
    weights = grads.view(grads.size(0), -1)
    weights = F.relu(weights)
    weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-5)
    cam = (weights.view(*weights.shape, *[1]*(feature.dim()-2)) * feature).sum(dim=1)
    return cam

def shape2bchw(x):
    if x.dim() == 4:
        return eop.rearrange(x, 'b c h w -> b () h (c w)')
    if x.dim() == 3:
        return eop.rearrange(x, 'b c h -> b () h c')
    if x.dim() == 2:
        return eop.rearrange(x, 'b c -> b () () c')
    
class set_record:
    def __init__(self, is_use:bool, model:nn.Module, sw:SummaryWriter, depth:int=6, step:int=0, save_cam:Optional[Dict]=None):
        self.hooks = []
        self.features = {}
        self.cam = {}
        self.train_mode = model.training
        self.grad = torch.is_grad_enabled()
        self.model = model
        self.model.eval()
        self.save_cam = save_cam
        torch.set_grad_enabled(True)
        def hook_forward(name, module, input, output):
            if not isinstance(output, torch.Tensor): return
            self.features[name] = output.clone().detach().cpu()
            sw.add_images(name.replace('.', '/'), shape2bchw(output).cpu(), global_step=step)
        def hook_backward(name, module, grad_in, grad_out):
            if not isinstance(grad_out[0], torch.Tensor): return
            grad = grad_out[0].clone().detach().cpu()
            feature = self.features.get(name, None)
            if feature is None: return
            if feature.shape == grad.shape:
                cam = shape2bchw(grad_cam(feature, grad)).transpose(-2, -1)
                if cam.max() <= 0.08: return
                self.cam[name] = cam
                sw.add_images('grad_cam/' + name.replace('.', '/'), cam.cpu(), global_step=step)
        if is_use:
            for name, module in model.named_modules():
                if len(name.split('.')) > depth: continue
                hook_fw = functools.partial(hook_forward, name)
                hook_bw = functools.partial(hook_backward, name)
                self.hooks.append(module.register_forward_hook(hook_fw))
                self.hooks.append(module.register_full_backward_hook(hook_bw))
    
    def __enter__(self):
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        [hook.remove() for hook in self.hooks]
        self.hooks.clear()
        self.features.clear()
        torch.set_grad_enabled(self.grad)
        self.model.train(self.train_mode)
        if self.save_cam is not None:
            self.save_cam.clear()
            self.save_cam.update(self.cam)
        self.cam.clear()

class record:
    def __init__(self, model:nn.Module, sw:SummaryWriter, depth:int=6, step:int=0, save_cam:Optional[Dict]=None):
        self.model = model
        self.sw = sw
        self.depth = depth
        self.step = step
        self.save_cam = save_cam
        self.record = None

    def __enter__(self):
        self.record = set_record(True, self.model, self.sw, self.depth, self.step, self.save_cam)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record.__exit__(exc_type, exc_val, exc_tb)

def cosmask(cos_freq, valley_length, mask, step):
    remainder = step % (cos_freq + valley_length)
    if remainder < valley_length:
        return torch.ones_like(mask)
    cos_progress = (1 - torch.cos((remainder - valley_length) * np.pi / cos_freq)) / 2
    prob = 0.5 * (1 - cos_progress)
    bernoulli_mask = torch.bernoulli(torch.empty_like(mask).fill_(prob))
    mask = mask * (1 - bernoulli_mask) 
    return 1 - mask

def plot_surface(data: torch.Tensor, path: str = None):
    Z = data.clone().detach().cpu().numpy()
    h, w = Z.shape
    X = np.arange(0, w, 1)
    Y = np.arange(0, h, 1)
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    color = cm.gray(Z)[:, :, :3]
    ax.plot_surface(X, Y, Z, facecolors=color)
    ax.set_zlim(0, 1)
    ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
    plt.savefig(path)