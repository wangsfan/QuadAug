from typing import Tuple
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST
from backbone.MNSITMLP import MNISTMLP
from backbone.ResNet import resnet18

from datasets.utils.continual_dataset import (ContinualDataset,
                                                  store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path

validation = False  # 是否需要验证集
class MyMNIST(MNIST):
    # 重写MNIST类，改变getitem函数
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
        self.not_aug_transform = transforms.ToTensor()  # 没有数据增强，直接转换为Tensor
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]

        # 这样做是为了与所有其他数据集保持一致
        # 返回一个PIL Image
        img = Image.fromarray(img.numpy(), mode='L')  # 实现array到image的转换
        original_img = self.not_aug_transform(img.copy())  # 将img浅复制并转为Tensor

        # 若存在数据增强
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # 如果数据存在logits属性
        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        # 返回 数据增强图像，图像标签，原始图像
        return img, target, original_img

class SeqMNIST(ContinualDataset):

    NAME = 'seq-mnist'
    SETTING = 'class-il'  # 类增量
    N_CLASSES_PER_TASK = 2  # 每个任务2个类别
    N_TASKS = 5  # 5个任务
    TRANSFORM = None  # 无数据增强

    def get_data_loaders(self):
        # 将图像转换为Tensor
        transform = transforms.ToTensor()
        # 下载数据
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=False, transform=transform)
        # 将训练集划分验证集
        if validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = MNIST(base_path() + 'MNIST',
                                 train=False, download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return resnet18(10)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_batch_size():
        return 64

    @staticmethod
    def get_minibatch_size():
        return SeqMNIST.get_batch_size()



