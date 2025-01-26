# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import torch
import random
import torch.nn as nn

import torchvision
# from matplotlib import pyplot as plt
from math import sqrt
from utils.scloss import SupConLoss

import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--eta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--lam', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--er_alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--er_beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--sscci_alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--crc_alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--csd_alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


# Define a function to randomly extract size different numbers from a 0-end sequence
def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


class QuadAug(ContinualModel):
    NAME = 'quadaug'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(QuadAug, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.new_img = None
        self.new_label = None
        self.criterion = SupConLoss()

    def phase_aug(self, img1, img2, lam, ratio=1.0):
        assert img1.shape == img2.shape
        c, h, w = img1.shape
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        img1_fft = np.fft.fft2(img1, axes=(1, 2))
        img2_fft = np.fft.fft2(img2, axes=(1, 2))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_pha = np.fft.fftshift(img1_pha, axes=(1, 2))
        img2_pha = np.fft.fftshift(img2_pha, axes=(1, 2))

        img1_pha_ = np.copy(img1_pha)
        img2_pha_ = np.copy(img2_pha)

        img1_pha[:, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img2_pha_[:, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_pha_[
                                                                                                 :,
                                                                                                 h_start:h_start + h_crop,
                                                                                                 w_start:w_start + w_crop]
        img2_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img1_pha_[:, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_pha_[
                                                                                                 :,
                                                                                                 h_start:h_start + h_crop,
                                                                                                 w_start:w_start + w_crop]

        img1_pha = np.fft.ifftshift(img1_pha, axes=(1, 2))
        img2_pha = np.fft.ifftshift(img2_pha, axes=(1, 2))

        img21 = img1_abs * (np.e ** (1j * img1_pha))
        img12 = img2_abs * (np.e ** (1j * img2_pha))
        img21 = np.real(np.fft.ifft2(img21, axes=(1, 2)))
        img12 = np.real(np.fft.ifft2(img12, axes=(1, 2)))

        return img21, img12

    def amplitude_aug(self, img1, img2, eta, ratio=1.0):

        assert img1.shape == img2.shape
        c, h, w = img1.shape
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        img1_fft = np.fft.fft2(img1, axes=(1, 2))
        img2_fft = np.fft.fft2(img2, axes=(1, 2))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_abs = np.fft.fftshift(img1_abs, axes=(1, 2))
        img2_abs = np.fft.fftshift(img2_abs, axes=(1, 2))

        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)

        img1_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            eta * img2_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - eta) * img1_abs_[
                                                                                                 :,
                                                                                                 h_start:h_start + h_crop,
                                                                                                 w_start:w_start + w_crop]
        img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            eta * img1_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - eta) * img2_abs_[
                                                                                                 :,
                                                                                                 h_start:h_start + h_crop,
                                                                                                 w_start:w_start + w_crop]

        img1_abs = np.fft.ifftshift(img1_abs, axes=(1, 2))
        img2_abs = np.fft.ifftshift(img2_abs, axes=(1, 2))

        img21 = img1_abs * (np.e ** (1j * img1_pha))
        img12 = img2_abs * (np.e ** (1j * img2_pha))
        img21 = np.real(np.fft.ifft2(img21, axes=(1, 2)))
        img12 = np.real(np.fft.ifft2(img12, axes=(1, 2)))

        return img21, img12

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def sscci_loss(self, f_a, f_b):
        # channel correlation matrix
        f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
        f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
        c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = self.off_diagonal(c).pow_(2).mean()
        loss = on_diag + 0.005 * off_diag
        return loss

    def observe(self, inputs, labels, not_aug_inputs):
        self.classifier_ad.to(self.device)
        self.detector.to(self.device)

        self.classifier_ad.train()
        self.detector.train()

        self.classifier_ad_optim.zero_grad()
        self.detector_optim.zero_grad()
        self.classifier_optim.zero_grad()
        self.opt.zero_grad()

        features = self.net(inputs)
        outputs = self.classifier(features)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():

            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_features = self.net(buf_inputs)
            buf_outputs = self.classifier(buf_features)

            loss += self.args.er_alpha * F.mse_loss(buf_outputs, buf_logits)
            abs_img_lst = []
            for i in range(inputs.shape[0]):
                abs_img_new_1, abs_img_new_2 = self.amplitude_aug(inputs[i].cpu(), buf_inputs[i].cpu(),
                                                                  self.args.eta)
                abs_img_lst.append(abs_img_new_2)

            pha_img_lst = []
            for i in range(inputs.shape[0]):
                pha_img_new_1, pha_img_new_2 = self.phase_aug(
                    buf_inputs[i].cpu(),
                    inputs[i].cpu(),
                    self.args.lam)
                pha_img_lst.append(pha_img_new_2)

            self.new_abs_img = torch.stack([torch.tensor(a) for a in abs_img_lst], 0)
            self.new_abs_img = self.new_abs_img.type(torch.FloatTensor).to(self.device)  # 转Float

            self.new_pha_img = torch.stack([torch.tensor(b) for b in pha_img_lst], 0)
            self.new_pha_img = self.new_pha_img.type(torch.FloatTensor).to(self.device)  # 转Float

            self.new_img = torch.cat((self.new_abs_img, self.new_pha_img))
            self.new_label = torch.cat((buf_labels[:inputs.shape[0]], labels))
            new_features = self.net(self.new_img)
            new_outputs = self.classifier(new_features)
            loss += self.args.er_beta * self.loss(torch.cat((buf_outputs, new_outputs)), torch.cat((buf_labels, self.new_label)))

            scores_sup = self.detector(new_features.detach())
            scores_inf = torch.ones_like(scores_sup) - scores_sup
            features_sup = new_features * scores_sup
            features_inf = new_features * scores_inf
            outputs_sup = self.classifier(features_sup)
            outputs_inf = self.classifier_ad(features_inf)
            loss_cls_sup = nn.CrossEntropyLoss()(outputs_sup, self.new_label)
            loss_cls_inf = nn.CrossEntropyLoss()(outputs_inf, self.new_label)

            loss += self.args.csd_alpha * (loss_cls_sup + loss_cls_inf)

            loss += self.args.sscci_alpha * self.sscci_loss(torch.cat((buf_features[:inputs.shape[0]], features)), new_features)

            cov = []
            for i in range(buf_features.shape[0]):
                feature_classwise = buf_features[i]
                feature_classwise = feature_classwise.reshape([512, -1])
                cov_channel = torch.matmul(feature_classwise, feature_classwise.T)
                cov_channel = F.normalize(cov_channel)
                cov.append(cov_channel)
            batch_cov = torch.stack(cov)

            abs_cov = []
            abs_features = new_features[:inputs.shape[0]]
            for i in range(abs_features.shape[0]):
                abs_feature_classwise = abs_features[i]
                abs_feature_classwise = abs_feature_classwise.reshape([512, -1])
                abs_cov_channel = torch.matmul(abs_feature_classwise, abs_feature_classwise.T)
                abs_cov_channel = F.normalize(abs_cov_channel)
                abs_cov.append(abs_cov_channel)
            abs_batch_cov = torch.stack(abs_cov)

            loss += self.args.crc_alpha * self.criterion(F.normalize(torch.reshape(batch_cov[:inputs.shape[0]],([batch_cov[:inputs.shape[0]].shape[0],-1])), dim=1), F.normalize(torch.reshape(abs_batch_cov,([abs_batch_cov.shape[0],-1])), dim=1), buf_labels[:inputs.shape[0]])

        loss.backward()
        self.opt.step()
        self.classifier_optim.step()
        self.classifier_ad_optim.step()

        # ---------------------------------- step2: update detector------------------------------
        if not self.buffer.is_empty():
            detector_features = self.net(torch.cat((inputs, self.new_img)))

            scores_sup = self.detector(detector_features.detach())
            scores_inf = torch.ones_like(scores_sup) - scores_sup
            features_sup = detector_features * scores_sup
            features_inf = detector_features * scores_inf
            outputs_sup = self.classifier(features_sup)
            outputs_inf = self.classifier_ad(features_inf)
            loss_cls_sup = nn.CrossEntropyLoss()(outputs_sup, torch.cat((labels, self.new_label)))
            loss_cls_inf = nn.CrossEntropyLoss()(outputs_inf, torch.cat((labels, self.new_label)))

            total_loss = loss_cls_sup - loss_cls_inf
            total_loss.backward()
            self.detector_optim.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)
        return loss.item()
