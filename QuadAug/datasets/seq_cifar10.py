from typing import Tuple

import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10
from datasets.utils.continual_dataset import (ContinualDataset,
                                                  store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from backbone.ResNet import resnet18
validation = False  # 是否需要验证集
class TCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
        self.root = root
        super(TCIFAR10, self).__init__(root, train, transform, target_transform, download)

class MyCIFAR10(CIFAR10):
    # 重写CIFAR10类，改变getitem函数
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]
        # 返回一个PIL image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

class SeqCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32),  # 将给定图像随机裁剪为不同的大小和高度比，再缩放为指定大小
             transforms.RandomHorizontalFlip(),  # 以给定概率随机水平旋转
             transforms.ToTensor(),  # 将图像转为Tensor
             transforms.Normalize((0.4914, 0.4822, 0.4465),  # 归一化处理
                                  (0.2470, 0.2435, 0.2615))])

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        if validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False,
                                    download=True, transform=test_transform)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SeqCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SeqCIFAR10.N_CLASSES_PER_TASK
                        * SeqCIFAR10.N_TASKS)


    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_epochs():
        return 10

    @staticmethod
    def get_batch_size():
        return 10

    @staticmethod
    def get_minibatch_size():
        return SeqCIFAR10.get_batch_size()