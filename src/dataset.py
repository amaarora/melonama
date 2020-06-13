import torch 
from PIL import Image
import albumentations
import numpy as np

class MelonamaDataset:
    def __init__(self, image_paths, targets, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets[idx]
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            'image': torch.tensor(image, dtype=torch.float), 
            'target': torch.tensor(target, dtype=torch.long)
        }
