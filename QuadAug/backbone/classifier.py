import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.layers = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        scores = self.layers(features)
        return scores


class Detector(nn.Module):
    def __init__(self, in_dim=512, num_classes=512, middle =1024, k = 308):
        super(Detector, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       score = self.bn(self.layers(f))
       z = torch.zeros_like(score)
       for _ in range(self.k):
           score = F.gumbel_softmax(score, dim=1, tau=0.5, hard=False)
           z = torch.maximum(score,z)
       return z