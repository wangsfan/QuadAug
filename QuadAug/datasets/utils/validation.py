import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

import os


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


class ValidationDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: np.ndarray,
                 transform: Optional[nn.Module] = None,
                  target_transform: Optional[nn.Module] = None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        # 返回验证集长度
        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            # 如果img是array 将array转化为PIL image
            if isinstance(img, np.ndarray):
                if np.max(img) < 2:  # 反归一化
                    img = Image.fromarray(np.uint8(img * 255))
                else:
                    img = Image.fromarray(img)
            else:
                img = Image.fromarray(img.numpy())
            # 若存在数据增强
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

#从训练集中划分验证集 val_perc：划分验证集的比例
def get_train_val(train: Dataset, test_transform: nn.Module,
                  dataset: str, val_perc: float = 0.1):
    dataset_length = train.data.shape[0] # 训练集长度
    directory = 'datasets/val_permutations/'
    create_if_not_exists(directory)  # 不存在路径则创建
    file_name = dataset + '.pt'
    if os.path.exists(directory + file_name):
        perm = torch.load(directory + file_name)
    else:
        perm = torch.randperm(dataset_length) # 将0~n-1(包括0和n-1)随机打乱后获得的数字序列
        torch.save(perm, directory + file_name)
    train.data = train.data[perm]  # 随机打乱训练集
    print('perm:',perm)
    train.targets = np.array(train.targets)[perm] # 随机打乱训练集标签
    # 训练集前val_perc * dataset_length大小为验证集
    test_dataset = ValidationDataset(train.data[:int(val_perc * dataset_length)],
                                     train.targets[:int(val_perc * dataset_length)],
                                     transform=test_transform)
    train.data = train.data[int(val_perc * dataset_length):]
    train.targets = train.targets[int(val_perc * dataset_length):]

    return train, test_dataset