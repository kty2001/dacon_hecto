import os
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.dataset import CarDataModule
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


def main(model, mode, data_dir, save_dir, ckpt_path):
    model = ClassficationModel(create_model(model=model), batch_size=CFG['BATCH_SIZE'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_loss',
        mode='max',
        dirpath= f'{save_dir}',
        filename= f'{model}-'+'{epoch:02d}-{val_epoch_loss:.2f}',
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
            callbacks=[checkpoint_callback, early_stopping],
        )

        transform = transforms.Compose([
            transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainer.fit(model, CarDataModule(data_dir, transform=transform, batch_size=64, image_size=256, mode='train'))
        trainer.test(model, CarDataModule(data_dir, transform=transform, batch_size=64, image_size=256, mode='train'))
    # else:
    #     trainer = L.Trainer(
    #         accelerator='gpus',
    #         devices='auto',
    #         precision='16-mixed'
    #     )
    #     model = ClassficationModel.load_from_checkpoint(ckpt, model=create_model(classification_model))
    #     pred_cls, img = trainer.predict(model, ImageNetDataModule(data, mode='predict'))[0]
    #     txt_path = '../dataset/folder_num_class_map.txt'
    #     classes_map = pd.read_table(txt_path, header=None, sep=' ')
    #     classes_map.columns = ['folder', 'number', 'classes']
        
    #     pred_label = classes_map['classes'][pred_cls.item()]
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, (800, 600))
    #     cv2.putText(
    #         img,
    #         f'Predicted class: "{pred_cls[0]}", Predicted label: "{pred_label}"',
    #         (50, 50),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.8,
    #         (0, 0, 0),
    #         2
    #     )
    #     cv2.imshow('Predicted output', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    seed_everything(CFG['SEED'])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data_dir", type=str, default="./data/train", help="Path to the data directory")
    parser.add_argument('--save_dir', type=str, default='./outputs/weights')
    parser.add_argument('--ckpt_path', type=str, default='./outputs/weights/')
    args = parser.parse_args()

    main(args.model, args.mode, args.data_dir, args.save_dir, args.ckpt_path)