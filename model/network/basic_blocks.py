
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        """
          Input:  x, [n, c, L]
          Output: ret, [n, c, L]
        """
        ret = self.conv(x)
        return ret

class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        h = x.size(2)
        split_size = int(h // 2**self.halving)
        z = x.split(split_size, 2)
        z = torch.cat([self.conv(_) for _ in z], 2)
        return F.leaky_relu(z, inplace=True)

class SequeBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SequeBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)

class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, part_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.part_num = part_num
        
        # MTB1
        conv3x1 = nn.Sequential(
                BasicConv1d(in_channels, hidden_dim, 3, padding=1), 
                nn.LeakyReLU(inplace=True), 
                BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, part_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)
        
        # MTB1
        conv3x3 = nn.Sequential(
                BasicConv1d(in_channels, hidden_dim, 3, padding=1), 
                nn.LeakyReLU(inplace=True), 
                BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, part_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

    def forward(self, x):
        """
          Input: x, [p, n, c, s]
        """
        p, n, c, s = x.size()
        feature = x.split(1, 0)
        x = x.view(-1, c, s)
        
        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
            for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1
        
        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
            for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3
        
        # Temporal Pooling
        ret = (feature3x1 + feature3x3).max(-1)[0]
        return ret

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs

class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)