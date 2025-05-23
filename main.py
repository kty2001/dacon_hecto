import os
import argparse

import numpy as np
import pandas as pd
import wandb

import torch
from torch import nn
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.dataset import CarDataModule
from src.model import create_model
from src.utils import CFG, seed_everything, multiclass_log_loss, submission


class ClassficationModel(L.LightningModule):
    def __init__(self,model, batch_size: int = 64):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.losses = []
        self.labels = []
        self.predictions = []
        self.probs = []

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self.model(inputs)
        loss = self.loss_fn(output, targets)

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
        acc = sum(labels == predictions) / len(labels)

        labels = labels.tolist()
        predictions = predictions.tolist()
        loss = sum(self.losses) / len(self.losses)

        # Logloss
        # answer_df = pd.DataFrame({'ID': range(len(labels)), 'label': labels})
        # submission_df = pd.DataFrame({'ID': range(len(predictions))})
        # num_classes = len(set(labels))
        # for i in range(num_classes):
        #     submission_df[i] = [1 if pred == i else 0 for pred in predictions]

        # try:
        #     log_loss_value = multiclass_log_loss(answer_df, submission_df)
        #     self.log('val_epoch_log_loss', log_loss_value)
        # except ValueError as e:
        #     print(f"Error calculating log loss: {e}")

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
        acc = sum(labels == predictions) / len(labels)

        labels = labels.tolist()
        predictions = predictions.tolist()
        loss = sum(self.losses) / len(self.losses)

        self.log('test_epoch_acc', acc)
        self.log('test_epoch_loss', loss)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()

    def predict_step(self, batch, batch_idx):
        inputs = batch
        output = self.model(inputs)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()

        return probs

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)


def main(m, mode, data_dir, save_dir, ckpt):
    model = ClassficationModel(model=create_model(m, num_classes=396),
                               batch_size=CFG['BATCH_SIZE'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_loss',
        mode='min',
        dirpath= f'{save_dir}',
        filename= f'{m}-'+'{epoch:02d}-{val_epoch_loss:.2f}',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_epoch_loss',
        mode='min',
        patience=10
    )
    wandb_logger = WandbLogger(project="Hecto")
    
    if mode == 'train':
        trainer = L.Trainer(
            accelerator='gpu',
            devices='auto',
            max_epochs=CFG['EPOCHS'],
            precision='16-mixed',
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping]
        )

        train_transform = transforms.Compose([
            transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainer.fit(model, CarDataModule(data_dir, transform=train_transform, batch_size=CFG['BATCH_SIZE'], mode='train'))
        trainer.test(model, CarDataModule(data_dir, transform=train_transform, batch_size=CFG['BATCH_SIZE'], mode='train'))
    
    elif mode == 'pred':
        trainer = L.Trainer(
            accelerator='gpu',
            devices=1,
            precision='16-mixed'
        )
        model = ClassficationModel.load_from_checkpoint(ckpt, model=create_model(m))
        pred_transform = transforms.Compose([
            transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        probs = trainer.predict(model, CarDataModule(data_dir, transform=pred_transform, mode='pred', batch_size=2048))
        probs = np.concatenate(probs, axis=0)
        
        file_name = f'resnet18'
        submission(probs=probs, file_name=file_name)

if __name__ == "__main__":
    seed_everything(CFG['SEED'])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    parser.add_argument('--save_dir', type=str, default='./outputs/weights')
    parser.add_argument('--ckpt', type=str, default='./outputs/weights/')
    args = parser.parse_args()

    main(args.model, args.mode, args.data_dir, args.save_dir, args.ckpt)

    wandb.finish()