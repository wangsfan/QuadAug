import importlib
import inspect
import os
from argparse import Namespace

from datasets.utils.continual_dataset import ContinualDataset


def get_all_datasets():
    return [model.split('.')[0] for model in os.listdir('/home/stu_mnt_point/wjl/QuadAug/datasets')
            if not model.find('__') > -1 and 'py' in model]


NAMES = {}
for dataset in get_all_datasets():
    mod = importlib.import_module('datasets.' + dataset)
    dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'ContinualDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c

    gcl_dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'GCLDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in gcl_dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c

def get_dataset(args: Namespace) -> ContinualDataset:
    # 创建和返回一个数据集
    assert args.dataset in NAMES
    return NAMES[args.dataset](args)