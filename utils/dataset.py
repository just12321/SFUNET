import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose, Resize
import re


from glob import glob
from pathlib import Path


class CIDataset(Dataset):
    def __init__(self, data_dir, augmentator, size, is_train=True, img_type="jpg",
                 pattern=lambda stem,train: re.sub(r'(training|test)', 'manual1', stem), verbose=False):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_type = img_type
        self.augmentator = augmentator
        self.image_paths = glob(f'{data_dir}/images/*.{img_type}')
        self.augmentator_ = is_train
        self.pattern = pattern
        self.verbose = verbose
        self.non_augmentator = Compose([Resize(*size), ToTensorV2(p=1)])

    def __getitem__(self, index):
        img_path = Path(self.image_paths[index])
        if self.verbose:print(img_path)
        mask_path = self.data_dir+'/manual/' + self.pattern(img_path.stem, self.is_train) + f'.{self.img_type}'
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert('L').point(lambda x:0 if x<128 else 1)) 
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1,1,3))
        if len(mask.shape) ==3:
            mask = mask[..., 0]
        res = self.augmentator(image = img, mask = mask) if self.augmentator_ else {'image':img, 'mask':mask}
        res = self.non_augmentator(**res)
        return res['image']/255., res['mask'][None, ...]
    
    def augment(self, v=True):
        self.augmentator_ = v

    def __len__(self):
        return len(self.image_paths)