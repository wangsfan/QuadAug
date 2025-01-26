from __future__ import print_function
import torch
import torch.nn as nn
from utils.conf import get_device
class CRDLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CRDLoss, self).__init__()
        self.temperature = temperature
        # self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        self.device = get_device()

    def forward(self, feature1, feature2, labels=None):
        batch_size = feature1.shape[0]  # 新样本+旧样本数
        # contiguous()保证一个Tensor是连续的  view(-1, 1)将一个不连续存储的张量重塑成一个列向量 [新样本+旧样本数，1]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        # 对两个张量Tensor进行逐元素的比较, 若相同位置的两个元素相同, 则返回True;
        mask = torch.eq(labels, labels.T).float().to(self.device)
        contrast_count = 1
        contrast_feature = feature1
        # 教师网络作为锚点
        anchor_feature = feature2
        anchor_count = 1

        # 计算两个特征相似度，并除以T
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # 为了数据稳定
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 计算同类
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



