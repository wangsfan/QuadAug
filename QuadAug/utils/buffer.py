from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

def icarl_replay(self, dataset, val_set_split=0):
    # 将当前任务数据合并为回放缓冲区
    # 选择性地划分回放缓冲区为验证集  val_set_split：验证集长度
    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()  # 以训练集大小初始化
        # 前buff_val_mask维度为True
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)
        # 判断数据类型，如果是Tensor使用torch.cat函数，不然使用np.concatenate
        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x):
                return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)  # 获取数据通道数
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)
        # 减少并合并训练集
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        # 减少并合并验证集
        if val_set_split > 0:
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    # num_seen_examples：可见样本数量
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

# 若知道任务数，每个任务划分buffer_portion_size个样本GEM
def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size

class Buffer:
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ('ring', 'reservoir')
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':  # 知道任务边界
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    # 将buffer中的参数都加载到device上
    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):  # hasattr用于判断对象是否具有指定的属性或方法
                # getattr获取某个类实例对象中指定属性的值
                # setattr修改类实例对象的属性值
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    # 取buffer的大小
    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    # 初始化获取到的Tensor
    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                     *attr.shape[1:]), dtype=typ, device=self.device))

    # 添加数据
    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)
        for i in range(examples.shape[0]):  # examples.shape[0]：样本数量
            index = reservoir(self.num_seen_examples, self.buffer_size)  # 以reservoir获取随机种子
            self.num_seen_examples += 1
            # 若buffer满了，则随机替换  若buffer不满，则依次填充
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        # 从buffer中选取size大小的一批随机样本
        # min()取当前buffer存在样本量  self.examples.shape[0]就是buffer_size
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])
        # numpy.random.choice(a, size=None, replace=True, p=None)
        # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        # replace:True表示可以取相同数字，False表示不可以取相同数字
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        # return (input,label,logit)
        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device),) + ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        # 根据index返回数据
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def get_all_data(self, transform: nn.Module = None) -> Tuple:
        # 返回buffer中所有数据
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def is_empty(self) -> bool:
        # 判断buffer是否为空
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        # 清空buffer
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


