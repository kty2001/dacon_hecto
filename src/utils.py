import os
import random

import numpy as np
import torch


CFG = {
    'IMG_SIZE': 256,
    'BATCH_SIZE': 192,
    'EPOCHS': 100,
    'LEARNING_RATE': 1e-4,
    'SEED' : 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
