from __future__ import print_function
import torch
import torch.nn as nn

from utils.conf import get_device
# 监督对抗学习损失
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = get_device()

    def forward(self, feature1, feature2, labels=None):
        # stack:用于连接两个相同大小的张量，并扩展维度   dim=1:根据相同特征维度进行拼接
        features = torch.stack([feature1, feature2], 1)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # 特征至少需要三维
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            # features1的维度：[c,n,m] c为样本数(新数据) n为长 m为高
            # 拼接features的维度：[c1+c2,2,n,m]
            # view：将提取出的特征图进行铺平，将特征图转换为一维向量 [c1+c2,2,n*m]
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]   # 样本数：新数据+buffer
        # 将一个不连续存储的张量重塑成一个列向量
        # labels:[样本数,1] torch.cat后 [样本数',1] contiguous().view [样本数',1] 样本数'：新数据+buffer
        labels = labels.contiguous().view(-1, 1)

        if labels.shape[0] != batch_size:  # label的样本数是否等于特征的样本数
            raise ValueError('Num of labels does not match num of features')
        # 对两个张量Tensor进行逐元素的比较, 若相同位置的两个元素相同, 则返回True;
        # mask：形状为[bsz，bsz]的对比掩码，如果样本j与样本i具有相同的类，则mask_{i，j}=1。可以是不对称的。
        # mask [样本数，样本数]
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature   # 目标特征点
        anchor_count = contrast_count
        # 先算特征相似度矩阵，再逐元素除以temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # 用于数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        # 根据mask大小初始化logits_mask 全部元素为1  先前对角线元素为0
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device), 0)
        # 获取对比编码矩阵 (同类为1 不同类及自身为0)
        mask = mask * logits_mask

        # 排除自身 分母部份
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask * log_prob 的每行都表示分子为正例，分母为全部的一个 lij，最后求和得到所有 lij 的损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()  # 求均值

        return loss