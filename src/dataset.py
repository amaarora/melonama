import torch 
from PIL import Image
import albumentations
import numpy as np
from color_constancy import color_constancy
import os

class MelonamaDataset:
    def __init__(self, image_paths, targets, augmentations=None, cc=True, meta_array=None):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.cc = cc
        self.meta_array = meta_array

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        if self.meta_array is not None:
            meta  = self.meta_array[idx]
        if self.cc: 
            image = color_constancy(image)
        target = self.targets[idx]
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            'image': torch.tensor(image, dtype=torch.float), 
            'target': torch.tensor(target, dtype=torch.long)
        } if self.meta_array is None else {
            'image': torch.tensor(image, dtype=torch.float), 
            'target': torch.tensor(target, dtype=torch.long),
            'meta': torch.tensor(meta, dtype=torch.float)
        }


class MelonamaTTADataset:
    """Only useful for TTA"""
    def __init__(self, image_paths, augmentations=None):
        self.image_paths = image_paths
        self.augmentations = augmentations 
        
    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        targets = torch.zeros(5)
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        images = self.augmentations(image)
        return {'image':images, 'target':targets}


class MelonamaMetaDataset:
    def __init__(self, df, data_dir, augmentations=None, cc=False, meta_features=['age_approx', 'sex']):
        images = df.image_name.tolist()
        self.image_paths = [os.path.join(data_dir, image_name+'.jpg') for image_name in images]
        self.targets = df.target
        self.augmentations = augmentations
        self.cc = cc
        self.df = df
        self.meta_features = meta_features

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        meta = np.array(
            self.df.iloc[idx][self.meta_features].values, dtype=np.float32)
        if self.cc: 
            image = color_constancy(image)
        target = self.targets[idx]
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            'image': torch.tensor(image, dtype=torch.float), 
            'meta': torch.tensor(meta, dtype=torch.float), 
            'target': torch.tensor(target, dtype=torch.long)
        }