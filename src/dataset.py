import os
import glob

from PIL import Image

from torch.utils.data import Dataset
import lightning as L


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            # 테스트셋: 라벨 없이 이미지 경로만 저장
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            # 학습셋: 클래스별 폴더 구조에서 라벨 추출
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label


class LightningDataModule(L.LightningDataModule):
    def __init__(self, data_dir, transform=None, batch_size=32, mode='train'):
        super().__init__()
        self.data_dir = data_dir # ./data/train
        self.transform = transform
        self.batch_size = batch_size
        self.mode = mode

    def setup(self, stage=None):
        if self.mode == 'train':
            data_list = glob.glob(os.path.join(self.data_dir, '*', '*.jpg'))
        if stage == 'fit' or stage is None:
            data_list = glob.glob(os.path.join(self.data_dir, '*', '*.jpg'))
        if stage == 'test' or stage is None:
            self.test_dataset = CustomImageDataset(self.test_dir, transform=self.transform, is_test=True)

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
        return L.pytorch.Trainer.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return L.pytorch.Trainer.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return L.pytorch.Trainer.DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def pred_dataloader(self):
        return L.pytorch.Trainer.DataLoader(self.test_dataset, batch_size=self.batch_size)