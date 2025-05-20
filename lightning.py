import os
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.dataset import LightningDataModule
from src.model import create_model
from src.utils import CFG, seed_everything


class ClassficationModel(L.LightningModule):
    def __init__(self,model, batch_size: int = 64):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.losses = []
        self.labels = []
        self.predictions = []

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss_fn(output, target)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss =  self.loss_fn(output, target)
        _, predictions = torch.max(output, 1)

        target_np = target.detach().cpu().numpy()
        predict_np = predictions.detach().cpu().numpy()
        
        self.losses.append(loss)
        self.labels.append(np.int16(target_np))
        self.predictions.append(np.int16(predict_np))
        self.log('valid_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        labels = np.concatenate(np.array(self.labels, dtype=object))
        predictions = np.concatenate(np.array(self.predictions, dtype=object))
        acc = sum(labels == predictions)/len(labels)

        labels = labels.tolist()
        predictions = predictions.tolist()
        loss = sum(self.losses)/len(self.losses)

        self.log('val_epoch_acc', acc)
        self.log('val_epoch_loss', loss)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss =  self.loss_fn(output, target)
        _, predictions = torch.max(output, 1)

        target_np = target.detach().cpu().numpy()
        predict_np = predictions.detach().cpu().numpy()
        
        self.losses.append(loss)
        self.labels.append(np.int16(target_np))
        self.predictions.append(np.int16(predict_np))
        self.log('test_loss', loss)
        return loss
    
    def on_test_epoch_end(self):
        labels = np.concatenate(np.array(self.labels, dtype = object))
        predictions = np.concatenate(np.array(self.predictions, dtype = object))
        acc = sum(labels == predictions)/len(labels)

        labels = labels.tolist()
        predictions = predictions.tolist()
        loss = sum(self.losses)/len(self.losses)

        self.log('test_epoch_acc', acc)
        self.log('test_epoch_loss', loss)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()

    def predict_step(self, batch, batch_idx):
        inputs, img = batch
        output = self.model(inputs)
        _, pred_cls = torch.max(output, 1)

        return pred_cls.detach().cpu().numpy(), img

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)


if __name__ == "__main__":
    seed_everything(CFG['SEED'])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data_dir", type=str, default="./data/train", help="Path to the data directory")
    parser.add_argument('--save_dir', type=str, default='./outputs/weights')
    parser.add_argument("--batch_size", type=int, default=CFG['BATCH_SIZE'], help="Batch size")
    parser.add_argument("--image_size", type=int, default=CFG['IMG_SIZE'], help="Image size")
    parser.add_argument("--epochs", type=int, default=CFG['EPOCHS'], help="Number of epochs")
    parser.add_argument('-c', '--ckpt_path', type=str, default='./outputs/weights/')
    args = parser.parse_args()
