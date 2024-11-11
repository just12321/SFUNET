import glob
import numpy as np
from PIL import Image
from pathlib import Path,WindowsPath
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def process(path:WindowsPath, despath:WindowsPath)->bool:
    img_dir = despath / 'images'
    gt_dir = despath / 'masks'
    img_path = path.parent / (path.stem.split('_')[0] + '.pgm')
    try:
        # np.save(img_dir / img_path.stem, np.array(Image.open(img_path)))
        # np.save(gt_dir / img_path.stem, np.array(Image.open(path)))
        Image.open(img_path).convert('RGB').save(img_dir / (img_path.stem+'.png'))
        Image.open(path).save(gt_dir / (img_path.stem+'.png'))
        return True
    except Exception as e:
        tqdm.write(f'{path}/{img_path} failed to process:{e}')
        return False

def divide(srcpath:str, despath:str):
    ratio = 0.8
    random.seed(42)
    gt_files = glob.glob(srcpath + '/*_gt.pgm')

    despath = Path(despath)
    train_dir = despath / 'train'
    val_dir = despath / 'val'

    (train_dir / 'images').mkdir(parents=True, exist_ok=True)
    (train_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (val_dir / 'images').mkdir(parents=True, exist_ok=True)
    (val_dir / 'masks').mkdir(parents=True, exist_ok=True)

    random.shuffle(gt_files)
    index = int(len(gt_files) * ratio)
    train_num, val_num = index, len(gt_files) - index
    valid_train, valid_val = 0, 0
    for item in tqdm(gt_files[:index],desc = "Processing training dataset"):valid_train+=1 if process(Path(item), train_dir) else 0
    for item in tqdm(gt_files[index:],desc = "Processing validation dataset"):valid_val+=1 if process(Path(item), val_dir) else 0
    print(f"{train_num+val_num} files found(train:{train_num};val:{val_num})")
    print(f"{valid_train} valid training files and {valid_val} valid validation files, total:{valid_train+valid_val}")

def _statistic(path):
    width,height=[],[]
    for dir in glob.glob(path + '/images/*'):
        img = np.load(dir)
        width.append(img.shape[1])
        height.append(img.shape[0])
    return width,height

def statistic(path):
    train_width,train_height=_statistic(path + '/train')
    val_width,val_height=_statistic(path + '/val')
    fig,ax=plt.subplots(2,2, figsize=(10, 8))

    ax[0, 0].hist(train_width,bins=100)
    ax[0, 0].set_title('Train width histogram')
    ax[0, 1].hist(train_height,bins=100)
    ax[0, 1].set_title('Train height histogram')

    ax[1, 0].hist(val_width,bins=100)
    ax[1, 0].set_title('Validation width histogram')
    ax[1, 1].hist(val_height,bins=100)
    ax[1, 1].set_title('Validation height histogram')
    plt.subplots_adjust(hspace = 0.5)
    plt.savefig(path + '/statistic.png')
