import math
import sys
from argparse import Namespace
from typing import Tuple
import os

import numpy as np
import torch
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.status import progress_bar
from copy import deepcopy
from utils.metrics import *
from torch.nn import functional as F
def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    # 给定输出张量、当前数据集、当前任务
    # 通过设置其他任务的响应为负无穷来屏蔽前者
    # 任务增量下：截取当前任务下的outputs
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    # 对过去每个任务评估模型准确率
    status = model.net.training
    model.net.eval()
    model.classifier.eval()
    # model.eval() 进入测试模式，负责改变batchnorm、dropout的工作方式，如在eval()模式下，dropout是不工作的
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():  # 用于指定在其内部的代码块中不进行梯度计算
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY: # 若不是类增量
                    outputs = model(inputs, k)    # k为任务标识符
                else:
                    outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]  # 获取总label数

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    model.classifier.train(status)
    # accs：全局准确率 accs_mask_classes：当前任务准确率
    return accs, accs_mask_classes

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # save args
    save_path = os.path.join(os.getcwd(), 'results', model.NAME + '-' + dataset.NAME + '-' + str(args.buffer_size))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'random_seed_record.txt'), 'a') as f:
        for arg in vars(args):
            f.write('{}:\t{}\n'.format(arg, getattr(args, arg)))
    # 模型训练
    model.net.to(model.device)
    model.classifier.to(model.device)
    results, results_mask_classes = [], []

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            model.classifier.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    all_accuracy_cls, all_accuracy_tsk = [], []
    all_forward_cls, all_forward_tsk = [], []
    all_backward_cls, all_backward_tsk = [], []
    all_forgetting_cls, all_forgetting_tsk = [], []
    all_acc_auc_cls, all_acc_auc_tsk = [], []
    for t in range(dataset.N_TASKS):
        model.net.train()  # 训练模式
        model.classifier.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        all_acc_auc_cls.append([])
        all_acc_auc_tsk.append([])

        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar(i, len(train_loader), epoch, t, loss)
                if model.NAME != 'icarl' and model.args.n_epochs == 1 and args.dataset != 'seq-imgnet1k' and model.NAME != 'scr':
                    if i % 5 == 0:
                        accs = evaluate(deepcopy(model), dataset)
                        all_acc_auc_cls[t].append(accs[0])
                        all_acc_auc_tsk[t].append(accs[1])



        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print(mean_acc)
        # print_mean_accuracy(mean_acc, t, dataset.SETTING)
        print('class-il:', accs[0], '\ntask-il:', accs[1])



        # record the results
        all_accuracy_cls.append(accs[0])
        all_accuracy_tsk.append(accs[1])
        # print the fwt, bwt, forgetting
        if model.NAME != 'icarl' and model.NAME != 'pnn' and model.NAME != 'scr':
            fwt = forward_transfer(results, random_results_class)
            fwt_mask_classes = forward_transfer(results_mask_classes, random_results_task)
            bwt = backward_transfer(results)
            bwt_mask_classes = backward_transfer(results_mask_classes)
            forget = forgetting(results)
            forget_mask_classes = forgetting(results_mask_classes)
            print('Forward: class-il: {}\ttask-il:{}'.format(fwt, fwt_mask_classes))
            print('Backward: class-il: {}\ttask-il:{}'.format(bwt, bwt_mask_classes))
            print('Forgetting: class-il: {}\ttask-il:{}'.format(forget, forget_mask_classes))

            # record the results
            all_forward_cls.append(fwt)
            all_forward_tsk.append(fwt_mask_classes)
            all_backward_cls.append(bwt)
            all_backward_tsk.append(bwt_mask_classes)
            all_forgetting_cls.append(forget)
            all_forgetting_tsk.append(forget_mask_classes)

    # record the results
    with open(os.path.join(save_path, 'random_seed_record.txt'), 'a') as f:
        f.write('\n== 1. Acc:\n==== 1.1. Class-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write(str(all_accuracy_cls[t]).strip('[').strip(']') + '\n')
        f.write('\n==== 1.2. Task-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write(str(all_accuracy_tsk[t]).strip('[').strip(']') + '\n')

        f.write('\n== 2. Forward:')
        f.write('\n==== 2.1. Class-IL:\n' + str(all_forward_cls).strip('[').strip(']'))
        f.write('\n==== 2.2. Task-IL:\n' + str(all_forward_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 3. Backward:')
        f.write('\n==== 3.1. Class-IL:\n' + str(all_backward_cls).strip('[').strip(']'))
        f.write('\n==== 3.2. Task-IL:\n' + str(all_backward_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 4. Forgetting:')
        f.write('\n==== 4.1. Class-IL:\n' + str(all_forgetting_cls).strip('[').strip(']'))
        f.write('\n==== 4.2. Task-IL:\n' + str(all_forgetting_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 5. Acc_auc:\n==== 5.1. Class-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write('\nTask {}:\n'.format(t + 1))
            avg_acc_cls, avg_acc_tsk = [], []
            for acc_cls, acc_tsk in zip(all_acc_auc_cls[t], all_acc_auc_tsk[t]):
                avg_acc_cls.append(np.mean(acc_cls))
                avg_acc_tsk.append(np.mean(acc_tsk))
                f.write(str(acc_cls).strip('[').strip(']') + ' - ' + str(np.mean(acc_cls)) + '\n')

        f.write('\nACC_AUC_cls = {}:\n'.format(np.mean(avg_acc_cls)))
        f.write('ACC_AUC_tsk = {}:\n'.format(np.mean(avg_acc_tsk)))
    return mean_acc[0],forget



# 输出平均准确率
def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    # 计算平均准确率
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
                  mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


