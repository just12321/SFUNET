import os
import importlib.util
import inspect
import json

from model.base import BaseModel

class ModelsManager:
    def __init__(self, root:str):
        self.root = root
        self.models = {}
        self._load_models_from_directory()

    def _load_models_from_directory(self):
        for subdir, _, files in os.walk(self.root):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(subdir, file)
                    self._import_models_from_file(filepath)

    def _import_models_from_file(self, filepath):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj is not BaseModel:
                self.models[name.upper()] = {
                    'class': obj,
                    'path': filepath
                }

    def get_model(self, model_name:str):
        model_name = model_name.upper()
        if model_name in self.models:
            return self.models[model_name]
        raise ValueError(f"Model {model_name} not found")

    def list_models(self):
        return list(self.models.keys())

    def save(self, save_path):
        save_data = {name: {'path': info['path']} for name, info in self.models.items()}
        with open(save_path, 'w') as f:
            json.dump(save_data, f)

    @classmethod
    def load(cls, load_path):
        with open(load_path, 'r') as f:
            loaded_data = json.load(f)
        
        instance = cls.__new__(cls)
        instance.root = None  # root is not necessary when loading from file
        instance.models = {}
        
        for name, info in loaded_data.items():
            instance._import_models_from_file(info['path'])
        
        return instance
