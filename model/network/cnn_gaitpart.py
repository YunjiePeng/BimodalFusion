import math
import torch
import torch.nn as nn
import numpy as np
import logging

from .basic_blocks import TemporalFeatureAggregator, SequeBlock, FocalConv2d

class GaitPart(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super(GaitPart, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        
        _set_in_channels = 1

        #---For CASIA-B---
        _set_channels = [math.ceil(hidden_dim/4), math.ceil(hidden_dim / 2), hidden_dim]
        self.layer1 = SequeBlock(FocalConv2d(_set_in_channels, _set_channels[0], 5, halving=0, padding=2))
        self.layer2 = SequeBlock(FocalConv2d(_set_channels[0], _set_channels[0], 3, halving=0, padding=1), True)
        self.layer3 = SequeBlock(FocalConv2d(_set_channels[0], _set_channels[1], 3, halving=2, padding=1))
        self.layer4 = SequeBlock(FocalConv2d(_set_channels[1], _set_channels[1], 3, halving=2, padding=1), True)
        self.layer5 = SequeBlock(FocalConv2d(_set_channels[1], _set_channels[2], 3, halving=3, padding=1))
        self.layer6 = SequeBlock(FocalConv2d(_set_channels[2], _set_channels[2], 3, halving=3, padding=1))

        #---For OU-MVLP---
        # _set_channels = [math.ceil(hidden_dim/8), math.ceil(hidden_dim/4), math.ceil(hidden_dim / 2), hidden_dim]
        # self.layer1 = SequeBlock(FocalConv2d(_set_in_channels, _set_channels[0], 5, halving=0, padding=2))
        # self.layer2 = SequeBlock(FocalConv2d(_set_channels[0], _set_channels[0], 3, halving=0, padding=1), True)
        # self.layer3 = SequeBlock(FocalConv2d(_set_channels[0], _set_channels[1], 3, halving=0, padding=1))
        # self.layer4 = SequeBlock(FocalConv2d(_set_channels[1], _set_channels[1], 3, halving=0, padding=1), True)
        # self.layer5 = SequeBlock(FocalConv2d(_set_channels[1], _set_channels[2], 3, halving=3, padding=1))
        # self.layer6 = SequeBlock(FocalConv2d(_set_channels[2], _set_channels[2], 3, halving=3, padding=1))
        # self.layer7 = SequeBlock(FocalConv2d(_set_channels[2], _set_channels[3], 3, halving=3, padding=1))
        # self.layer8 = SequeBlock(FocalConv2d(_set_channels[3], _set_channels[3], 3, halving=3, padding=1))
        
        self.TFA = TemporalFeatureAggregator(_set_channels[-1])
        self.FCs = nn.Parameter(nn.init.xavier_uniform_(
                        torch.zeros(16, hidden_dim, out_channels)))
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2)
        del silho

        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.layer5(x)
        x = self.layer6(x) # [n, s, c, h, w]

        #---For OU-MVLP---
        # x = self.layer7(x)
        # x = self.layer8(x)
        
        # HP: horizontal pooling
        n, s, c, h, w = x.size()
        hp_feature = x.view(n, s, c, 16, -1)
        hp_feature = hp_feature.max(-1)[0] + hp_feature.mean(-1) # [n, s, c, p]
        hp_feature = hp_feature.permute(3, 0, 2, 1).contiguous() # [p, n, c, s]
        
        tfa_feature = self.TFA(hp_feature) # [p, n, c]
        feature = tfa_feature.matmul(self.FCs) # [p, n, c]

        tfa_feature = tfa_feature.permute(1, 0, 2).contiguous() # [n, p, c]
        feature = feature.permute(1, 0, 2).contiguous() # [n, p, c]
        
        return tfa_feature, feature
