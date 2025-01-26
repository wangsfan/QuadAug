import random
import torch as torch
import numpy as np
import os
def get_device() -> torch.device:
    #如果GPU空闲返回GPU 不然返回CPU
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except:
        pass

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# def set_random_seed(seed: int) -> None:
#     """
#     Sets the seeds at a certain value.
#     :param seed: the value to be set
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return '/home/stu_mnt_point/wjl/QuadAug/'
    #return '/Users/wujialu/PycharmProjects/DER/'

def base_path_dataset() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return '/home/stu_mnt_point/wjl/QuadAug/data/'
    #return '/Users/wujialu/PycharmProjects/DER/gcl_datasets/'