import os
# os.chdir(r'/git/continual_learning/mammoth')
import importlib
import numpy as np
import torch

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed

def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    # parser.add_argument('--weight_dist',type=str, required=True,
    #                     help='what type of weight distribution assigned to classes to sample (unif or longtail)')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    print(args)

    if args.load_best_args:
        parser.add_argument('--experiment_id', type=str, default='best_args')
        parser.add_argument('--tiny_imagenet_path', type=str, default='data')

        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset]['sgd'][args.weight_dist]
            else:
                best = best_args[args.dataset]['sgd']
        else:
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset][args.model][args.weight_dist]
            else:
                best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

        print(args)



    if args.model == 'mer':
        setattr(args, 'batch_size', 1)


    if args.seed is not None:
        set_random_seed(args.seed)
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    if isinstance(dataset, ContinualDataset):
        acc,forget = train(model, dataset, args)
        print('=' * 100)
        print('----------- Avg_Acc {} -----------'.format(acc))
        print('----------- Avg_Forget {} -----------'.format(forget))
        print('=' * 100)
    else:
        assert not hasattr(model, 'end_task')
        acc = ctrain(args)
        print('=' * 100)
        print('----------- Avg_Acc {} -----------'.format(acc))
        print('=' * 100)

if __name__ == '__main__':
    for i in range(10):
        main()