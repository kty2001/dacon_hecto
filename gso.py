# main_ocr_kfold.py

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import easyocr

# ----- 설정 -----
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 128,
    'EPOCHS': 1,
    'LEARNING_RATE': 1e-4,
    'SEED': 42,
    'NUM_FOLDS': 2
}

random.seed(CFG['SEED'])
os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- OCR 설정 -----
ocr_reader = easyocr.Reader(['en'])

# 실제 클래스 이름 기반 힌트 키워드 → 주요 차종
OCR_HINT_MAP = {
    'bmw': 'BMW',
    'benz': '벤츠',
    'kia': 'K',
    'hyundai': '그랜저',
    'avante': '아반떼',
    'grandeur': '그랜저',
    'sorento': '쏘렌토',
    'genesis': 'G80',
    'audi': 'A6',
    'peugeot': '3008',
    'jeep': '랭글러',
    'porsche': '911',
    'volvo': 'XC60',
    'lexus': 'ES300h',
    'cadillac': 'CT6',
    'malibu': '말리부',
    'carnival': '카니발',
    'spark': '스파크'
    # 필요한 경우 추가 확장 가능
}

def apply_ocr_boost(image_path, probs, class_names, boost_value=0.10):
    result = ocr_reader.readtext(image_path, detail=0)
    detected_text = ' '.join(result).lower()
    boosted_probs = probs.clone()

    # 1. 브랜드 단서 먼저 추론
    matched_brands = []
    for keyword, class_keyword in OCR_HINT_MAP.items():
        if keyword in detected_text:
            matched_brands.append(class_keyword)

    # 2. 브랜드만 매칭되는 경우: class 이름 중 브랜드명 포함된 것만 boost
    if matched_brands:
        for i, name in enumerate(class_names):
            if any(brand in name for brand in matched_brands):
                boosted_probs[i] += boost_value / 2  # 가벼운 boost

    # 3. 브랜드 + 시리즈가 동시에 텍스트에 있다면 더 강하게 boost
    for i, name in enumerate(class_names):
        tokens = name.lower().split('_')
        if all(token in detected_text for token in tokens[:2]):  # 브랜드 + 시리즈명
            boosted_probs[i] += boost_value  # 강한 boost

    boosted_probs /= boosted_probs.sum()
    return boosted_probs

# ----- 데이터셋 정의 -----
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            self.img_list = sorted(os.listdir(root_dir))
            for fname in self.img_list:
                if fname.lower().endswith('.jpg'):
                    self.samples.append((fname,))
        else:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                for fname in os.listdir(os.path.join(root_dir, cls_name)):
                    if fname.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(cls_name, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            fname = self.samples[idx][0]
            image = Image.open(os.path.join(self.root_dir, fname)).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, fname
        else:
            img_path, label = self.samples[idx]
            image = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

# ----- 모델 정의 -----
class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# ----- 트랜스폼 -----
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(CFG['IMG_SIZE'], scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----- 전체 파이프라인 -----
def train_and_infer(train_root, test_root, submission_csv):
    full_dataset = CustomImageDataset(train_root, transform=None)
    targets = [label for _, label in full_dataset.samples]
    class_names = sorted(os.listdir(train_root))

    test_dataset = CustomImageDataset(test_root, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
    all_test_probs = []

    skf = StratifiedKFold(n_splits=CFG['NUM_FOLDS'], shuffle=True, random_state=CFG['SEED'])

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset.samples, targets)):
        print(f"\n📦 Fold {fold+1}/{CFG['NUM_FOLDS']}")

        train_dataset = Subset(CustomImageDataset(train_root, transform=train_transform), train_idx)
        val_dataset = Subset(CustomImageDataset(train_root, transform=val_transform), val_idx)
        train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

        model = BaseModel(num_classes=len(class_names)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(CFG['EPOCHS']):
            for images, labels in tqdm(train_loader, desc=f"[Fold {fold+1}][Epoch {epoch+1}]"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Validation logloss 측정
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))
        print(f"📊 Fold {fold+1} Validation LogLoss: {val_logloss:.4f}")

        # 테스트 예측
        fold_test_probs = []
        with torch.no_grad():
            for images, fnames in tqdm(test_loader, desc=f"[Fold {fold+1}] Inference"):
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                for i, prob in enumerate(probs):
                    path = os.path.join(test_root, fnames[i])
                    boosted_prob = apply_ocr_boost(path, prob, class_names)
                    fold_test_probs.append(boosted_prob.cpu().numpy())
        all_test_probs.append(fold_test_probs)

    # 평균 앙상블
    final_probs = np.mean(all_test_probs, axis=0)
    pred_df = pd.DataFrame(final_probs, columns=class_names)

    submission = pd.read_csv(submission_csv)
    submission[class_names] = pred_df[class_names].values
    submission.to_csv('submission_ocr_kfold.csv', index=False, encoding='utf-8-sig')


train_and_infer('./dataset/train', './dataset/test', './dataset/sample_submission.csv')