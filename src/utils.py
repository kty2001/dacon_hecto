import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import torch


CFG = {
    'IMG_SIZE': 256,
    'BATCH_SIZE': 384,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-4,
    'SEED' : 42
}

CLASS_DICT = {}
with open('./data/class_dict.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        class_name, class_id = line.strip().split()
        CLASS_DICT[class_id] = class_name


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def multiclass_log_loss(answer_df, submission_df):
    class_list = sorted(answer_df['label'].unique())
    
    if submission_df.shape[0] != answer_df.shape[0]:
        raise ValueError("submission_df 행 개수가 answer_df와 일치하지 않습니다.")

    submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)
    answer_df = answer_df.sort_values(by='ID').reset_index(drop=True)

    if not all(answer_df['ID'] == submission_df['ID']):
        raise ValueError("ID가 정렬되지 않았거나 불일치합니다.")
    
    missing_cols = [col for col in class_list if col not in submission_df.columns]
    if missing_cols:
        raise ValueError(f"클래스 컬럼 누락: {missing_cols}")
    
    if submission_df[class_list].isnull().any().any():
        raise ValueError("NaN 포함됨")
    for col in class_list:
        if not ((submission_df[col] >= 0) & (submission_df[col] <= 1)).all():
            raise ValueError(f"{col}의 확률값이 0~1 범위 초과")

    true_labels = answer_df['label'].tolist()
    true_idx = [class_list.index(lbl) for lbl in true_labels]

    probs = submission_df[class_list].values
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = np.clip(probs, 1e-15, 1 - 1e-15)

    return log_loss(true_idx, y_pred, labels=list(range(len(class_list))))

def submission(probs, file_name):
    results = []
    for prob in probs:
        result = {}
        result = {CLASS_DICT[str(i)]: prob[i].item() for i in range(len(CLASS_DICT))}
        results.append(result)
    
    pred = pd.DataFrame(results)

    submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')

    class_columns = submission.columns[1:]
    pred = pred[class_columns]
    submission[class_columns] = pred.values
    submission.to_csv(f'./outputs/submissions/{file_name}.csv', index=False, encoding='utf-8-sig')
