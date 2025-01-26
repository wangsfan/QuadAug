# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from backbone.network import network
from backbone.classifier import Detector,Classifier
class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()
        self.numclass = 100
        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()

        self.dim = 512
        self.k = 308
        # networks
        self.encoder = backbone
        self.detector = Detector(in_dim=self.dim, num_classes=self.dim, middle=4 * self.dim, k=self.k)
        self.classifier = Classifier(self.dim, 100).to(self.device)
        self.classifier_ad = Classifier(self.dim, 100).to(self.device)
        # optimizers
        self.encoder_optim =SGD(self.net.parameters(), lr=self.args.lr)
        self.classifier_optim = SGD(self.classifier.parameters(), lr=self.args.lr)
        self.classifier_ad_optim = SGD(self.classifier_ad.parameters(), lr=self.args.lr)
        self.detector_optim = SGD(self.detector.parameters(), lr=self.args.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        out = self.net(x)
        return self.classifier(out)


    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass