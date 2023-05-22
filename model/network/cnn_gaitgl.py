import math
import torch
import torch.nn as nn
import numpy as np
import logging

from .basic_blocks import BasicConv3d, GLConv, GeMHPP

class GaitGL(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super(GaitGL, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        
        _set_in_channels = 1

        #---For CASIA-B---
        class_num = 73
        in_c = [math.ceil(hidden_dim/4), math.ceil(hidden_dim / 2), hidden_dim]
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )
        self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        #---For OU-MVLP---
        # class_num = 5153
        # in_c =  [math.ceil(hidden_dim/8), math.ceil(hidden_dim/4), math.ceil(hidden_dim / 2), hidden_dim]
        # self.conv3d = nn.Sequential(
        #     BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
        #                 stride=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.LeakyReLU(inplace=True),
        #     BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
        #                 stride=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.LeakyReLU(inplace=True),
        # )
        # self.LTA = nn.Sequential(
        #     BasicConv3d(in_c[0], in_c[0], kernel_size=(
        #         3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
        #     nn.LeakyReLU(inplace=True)
        # )

        # self.GLConvA0 = nn.Sequential(
        #     GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
        #         3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        #     GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
        #         3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        # )
        # self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # self.GLConvA1 = nn.Sequential(
        #     GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
        #         3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        #     GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
        #         3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        # )
        # self.GLConvB2 = nn.Sequential(
        #     GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
        #         3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        #     GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
        #         3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        # )

        self.HPP = GeMHPP()
        self.Head0 = nn.ParameterList([
                        nn.Parameter(
                            nn.init.xavier_uniform_(
                                torch.zeros(64, in_c[-1], in_c[-1])))])
        self.Bn = nn.BatchNorm1d(in_c[-1])
        self.Head1 = nn.ParameterList([
                        nn.Parameter(
                            nn.init.xavier_uniform_(
                                torch.zeros(64, in_c[-1], class_num)))])

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 2)
        else:
            _tmp = [
                torch.max(x[:, :, self.batch_frame[i]:self.batch_frame[i + 1], :, :], 2)
                for i in range(len(self.batch_frame) - 1)
            ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

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

        n, s, c, h, w = x.size()
        x = x.view(n, 1, s, h, w) # [n, t, s, h, w]
        
        if s < 3:
            repeat = 3 if s == 1 else 2
            x = x.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(x)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        # print("outs:{}".format(outs.size()))

        outs = self.frame_max(outs)[0] # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]
        outs = outs.permute(2, 0, 1).contiguous()  # [p, n, c]

        # print("outs:{}, head0:{}".format(outs.size(), self.Head0[0].size()))

        gait = outs.matmul(self.Head0[0]) # [p, n, c]
        gait = gait.permute(1, 2, 0).contiguous()  # [n, c, p]
        bnft = self.Bn(gait)  # [n, c, p]
        logi = bnft.permute(2, 0, 1).contiguous().matmul(self.Head1[0])  # [p, n, c]

        gait = gait.permute(0, 2, 1).contiguous()  # [n, p, c]
        bnft = bnft.permute(0, 2, 1).contiguous()  # [n, p, c]
        # logi = logi.permute(1, 0, 2).contiguous()  # [n, p, c]

        # print("outs:{}, gait:{}, bnft:{}, logi:{}".format(outs.size(), gait.size(), bnft.size(), logi.size()))
        
        return bnft, logi
