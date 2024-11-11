import importlib
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from types import ModuleType
from utils.dataset import CIDataset
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image, make_grid

from utils.utils import binary, record


def predict(model):
    model.eval()
    model.clear_pre()
    img, mask = next(iter(val_loader))
    img, mask = img.to(device), mask.to(device)
    pred = binary(model(img))
    cmpmask = torch.zeros_like(pred).repeat([1, 3, 1, 1])
    true_positive = (pred == 1) & (mask == 1)  
    false_positive = (pred == 1) & (mask == 0)  
    false_negative = (pred == 0) & (mask == 1)  
    cmpmask = torch.where(true_positive, torch.tensor([1, 1, 1], device=device).view(1, 3, 1, 1), cmpmask)  
    cmpmask = torch.where(false_positive, torch.tensor([1, 0, 0], device=device).view(1, 3, 1, 1), cmpmask)  
    cmpmask = torch.where(false_negative, torch.tensor([0, 1, 0], device=device).view(1, 3, 1, 1), cmpmask)  

    return pred, cmpmask

def import_module_from_file(module_name: str, file_path: str) -> ModuleType:
    """
    Import a module given its file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def traverse_and_import(base_dir: str, save_dir: str, filter_func: callable = None) -> dict:
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
                for i in range(5):
                    checkpoint_path = os.path.join(root, '..', 'checkpoint', f'fold_{i}_best.pt')
                    net.load_state_dict(torch.load(checkpoint_path)['model'])
                    pred, predcmp = predict(net)
                    save_image(make_grid(pred, 1), f'{save_dir}/{model_name}_{iteration}_{i}.png')
                    save_image(make_grid(predcmp, 1), f'{save_dir}/{model_name}_{iteration}_{i}_mark.png')
                del net

path1 = r'./dataset/Database134/val'
valset = CIDataset(path1, augmentator=None, size=(300, 300), is_train=False, img_type='png', pattern=lambda stem, _:stem)

val_loader = DataLoader(valset, batch_size=3, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_directory = 'exp/KFold'
save_directory = 'vis'
filter_func = lambda model_name, iteration: 'DCA1' in iteration.upper()
traverse_and_import(base_directory, save_directory, filter_func)
exit(0)
