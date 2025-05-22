import os
import glob
from pathlib import Path

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as L

from src.utils import CFG, seed_everything

seed_everything(CFG['SEED'])

class CarDataModule(L.LightningDataModule):
    def __init__(self, data_dir, transform=None, batch_size=32, mode='train'):
        super().__init__()
        self.data_dir = data_dir # ./data/train
        self.transform = transform
        self.batch_size = batch_size
        self.mode = mode


    def setup(self, stage=None):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).")
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.mode == 'train':
            data_list = glob.glob(os.path.join(self.data_dir, '*', '*.jpg'))
            target_list = [os.path.basename(os.path.dirname(p)) for p in data_list]
            
            train_x, val_x, train_y, val_y = train_test_split(data_list, target_list, test_size=0.2, stratify=target_list, random_state=42)
            val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.5, stratify=val_y, random_state=42)

            train_data = [(x, y) for x, y in zip(train_x, train_y)]
            val_data = [(x, y) for x, y in zip(val_x, val_y)]
            test_data = [(x, y) for x, y in zip(test_x, test_y)]
        else:
            pred_data = test_data[0]

        if stage == 'fit':
            self.train_dataset = train_data
            self.val_dataset = val_data
        if stage == 'test':
            self.test_dataset = test_data
        if stage == 'predict':
            self.pred_dataset = pred_data

    def _train_collate_fn(self, batch):
        images, labels = zip(*batch)
        images = [self.transform(Image.open(img).convert('RGB')) for img in images]
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
        return torch.stack(images), labels
    
    def _predict_collate_fn(self, batch):
        img = batch[0]
        input = self.transform(img).unsqueeze(0)
        return input, np.array(img)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_per_device, shuffle=True, collate_fn=self._train_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._train_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._train_collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._predict_collate_fn) 