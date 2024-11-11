import importlib
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from utils.losses import IOU_loss
from utils.metrics import Accuracy, Dice, Iou, Recall, mIoU
from types import ModuleType
from tqdm import tqdm
from model.sfunet import SFUnet
from utils.Accumulator import Accumulators
from utils.dataset import CIDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F

from utils.utils import overlay_mask, record

ms = {
    'Acc': Accuracy,
    'Iou': Iou,
    'F1': Dice,
    'Recall': Recall,
    'mIoU': mIoU,
}

def metrics(model):
    metrics_acc = Accumulators()
    for idx, (img, mask) in tqdm(enumerate(val_loader)):
        model.clear_pre()
        img, mask = img.to(device), mask.to(device)
        metrics = model.metrics(img, mask, metrics=ms)
        metrics_acc.update(metrics)
    return metrics_acc.items()

def import_module_from_file(module_name: str, file_path: str) -> ModuleType:
    """
    Import a module given its file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def traverse_and_import(base_dir: str, filter_func: callable = None) -> dict:
    """
    Traverse the directory and dynamically import model and args from model.py.
    """
    for root, dirs, files in os.walk(base_dir):
        if 'model.py' in files:
            # Extract model name and iteration from the path
            parts = root.split(os.sep)
            model_name = parts[-3]
            iteration = parts[-2]

            if filter_func and not filter_func(model_name, iteration):
                continue


            # Construct the full path to model.py
            model_path = os.path.join(root, 'model.py')
            
            # Dynamically import Name from model.py
            model_py_module = import_module_from_file('model.py', model_path)
            model_name_in_file = model_py_module.modelName            
            args = model_py_module.args
            # Construct the full path to the specific model file
            specific_model_path = os.path.join(root, f'{model_name_in_file}.py')
            
            if os.path.exists(specific_model_path):
                # Dynamically import model and args from the specific model file
                specific_model_module = import_module_from_file(model_name_in_file, specific_model_path)
                model = getattr(specific_model_module, model_name_in_file)
            
                print(f"Model Name: {model_name}, Iteration: {iteration}")
                net = model(**args).to(device)
                net.eval()
                test_accs = Accumulators()
                for i in range(5):
                    checkpoint_path = os.path.join(root, '..', 'checkpoint', f'fold_{i}_best.pt')
                    net.load_state_dict(torch.load(checkpoint_path)['model'])
                    res = metrics(net)
                    test_accs.update(res)
                print('Test metrics:', test_accs.items())
                del net

path = r'./dataset/Database134/val'
valset = CIDataset(path, augmentator=None, size=(300, 300), is_train=False, img_type='png', pattern=lambda stem, _:stem)

val_loader = DataLoader(valset, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_directory = 'exp/KFold'
filter_func = lambda model_name, iteration: 'sfunet' in model_name.lower() and 'DCA1_300' in iteration

# traverse_and_import(base_directory, filter_func)
# exit(0)

sw = SummaryWriter('exp/TEST/vision')
model = SFUnet(3, 1, num_heads=2).to(device)
model.load_state_dict(torch.load(r'exp/KFold/info/SFUnet/DCA1_300_300_000/checkpoint/fold_0_best.pt')['model'])
model.eval()

# Visualize the network
print("Waiting for forward...")
cam = {}
with record(model, sw, depth=9, save_cam=cam):
    img, mask = next(iter(val_loader))
    img, mask = img.to(device), mask.to(device)
    pred = model(img)
    print("Waiting for backward...")
    loss = IOU_loss(pred, mask.float())
    loss.backward()
for k, v in cam.items():
    try:
        grad_cam = F.interpolate(v, size=img.shape[2:], mode='bicubic', align_corners=False).view(*img.shape[2:])
        heat_map = overlay_mask(img.squeeze(0).permute(1, 2, 0).cpu(), grad_cam, alpha=0.5)
        sw.add_image(f"heatmap/{k}", heat_map, 0, dataformats='HWC')
    except Exception as e:
        print(f"{k} failed:{e}")
    finally:
        continue
cam.clear()
sw.close()