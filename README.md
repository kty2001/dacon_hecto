# dacon_hecto
## Car Classification

### requirements
mine
```bash
conda create -n hecto1 python=3.12 -y
conda activate hecto1
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pandas tqdm scikit-learn wandb git -y
conda install lightning -c conda-forge -y

```

coworker's
```bash
conda create -n hecto1 python=3.11 -y
conda activate hecto1
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pandas tqdm scikit-learn git -y
conda install lightning -c conda-forge -y
conda install easyocr -c conda-forge -y

```