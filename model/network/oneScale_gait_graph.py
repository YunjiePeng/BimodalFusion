import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .graph import SpatialGraph
from .scn import SCN

import logging

class OSGG(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_id,
                 graph_cfg,
                 edge_importance_weighting=True,
                 **kwargs):
        super().__init__()

        # load spatial graph
        self.graph = SpatialGraph(**graph_cfg)
        A1 = torch.tensor(self.graph.get_adjacency(semantic_level=0), dtype=torch.float32, requires_grad=False)
        A2 = torch.tensor(self.graph.get_adjacency(semantic_level=0), dtype=torch.float32, requires_grad=False)
        A3 = torch.tensor(self.graph.get_adjacency(semantic_level=0), dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('A1', A1)
        self.register_buffer('A2', A2)
        self.register_buffer('A3', A3)

        # build networks
        spatial_kernel_size = self.graph.num_A
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        #---For CASIA-B---
        out_channel_list = [int(out_channels / 4), int(out_channels / 2)]
        self.st_gcn_networks_A1 = nn.ModuleList((
            st_gcn_block(in_channels, out_channel_list[0], kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(out_channel_list[0], out_channel_list[0], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[0], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channels, kernel_size, 1, **kwargs),
            st_gcn_block(out_channels, out_channels, kernel_size, 1, **kwargs),
        ))

        self.st_gcn_networks_A2 = nn.ModuleList((
            st_gcn_block(in_channels, out_channel_list[0], kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(out_channel_list[0], out_channel_list[0], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[0], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channels, kernel_size, 1, **kwargs),
            st_gcn_block(out_channels, out_channels, kernel_size, 1, **kwargs),
        ))
        self.st_gcn_networks_A3 = nn.ModuleList((
            st_gcn_block(in_channels, out_channel_list[0], kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(out_channel_list[0], out_channel_list[0], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[0], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channels, kernel_size, 1, **kwargs),
            st_gcn_block(out_channels, out_channels, kernel_size, 1, **kwargs),
        ))
 
        self.edge_importance_weighting = edge_importance_weighting
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_A1 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A1.size()))
                for i in self.st_gcn_networks_A1
            ])
            self.edge_importance_A2 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A2.size()))
                for i in self.st_gcn_networks_A2
            ])
            self.edge_importance_A3 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A3.size()))
                for i in self.st_gcn_networks_A3
            ])
        else:
            self.edge_importance_A1 = [1] * len(self.st_gcn_networks_A1)
            self.edge_importance_A2 = [1] * len(self.st_gcn_networks_A2)
            self.edge_importance_A3 = [1] * len(self.st_gcn_networks_A3)

        self.fc = nn.Linear(out_channels, out_channels)
        self.bn_neck = nn.BatchNorm1d(out_channels)
        self.encoder_cls = nn.Linear(out_channels, num_id, bias=False)

    def semantic_pooling(self, x):
        cur_node_num = x.size()[-1]
        half_x_1, half_x_2 = torch.split(x, int(cur_node_num / 2), dim=-1)
        x_sp = torch.add(half_x_1, half_x_2) / 2
        return x_sp

    def forward(self, x, batch_frame=None):
        if batch_frame is not None: # when testing, the input is the whole sequence.
            # Parameter batch_frame is set to handle this problem.
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < x.size(1):
                x = x[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()

        '''N - the number of videos.
           T - the number of frames in one video.
           V - the number of keypoints.
           C - the number of features for one keypoint.
        '''
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, C, T, V)

        y = x
        z = y
        if self.edge_importance_weighting:
            for gcn_A1, importance_A1, gcn_A2, importance_A2, gcn_A3, importance_A3 in zip(self.st_gcn_networks_A1, self.edge_importance_A1, self.st_gcn_networks_A2, self.edge_importance_A2, self.st_gcn_networks_A3, self.edge_importance_A3):
                x, _ = gcn_A1(x, self.A1 * importance_A1)
                y, _ = gcn_A2(y, self.A2 * importance_A2)
                z, _ = gcn_A3(z, self.A3 * importance_A3)

                y = torch.add(y, x)
                y_sp = self.semantic_pooling(y)
                z = torch.add(z, y)
        else:
            for gcn_A1, gcn_A2, gcn_A3 in zip(self.st_gcn_networks_A1, self.st_gcn_networks_A2, self.st_gcn_networks_A3):
                x, _ = gcn_A1(x, self.A1)
                y, _ = gcn_A2(y, self.A2)
                z, _ = gcn_A3(z, self.A3)

                y = torch.add(y, x)
                z = torch.add(z, y)

        # print("x_sp:{}".format(x_sp.size()))
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, T*V, C)

        y = F.avg_pool2d(y, y.size()[2:])
        N, C, T, V = y.size()
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(N, T*V, C)

        z = F.avg_pool2d(z, z.size()[2:])
        N, C, T, V = z.size()
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(N, T*V, C)

        z_fc = self.fc(z.view(N, -1))
        bn_z_fc = self.bn_neck(z_fc)
        z_cls_score = self.encoder_cls(bn_z_fc)

        z_fc = z_fc.unsqueeze(1).contiguous()
        return x, y, z, z_fc, z_cls_score

class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size, i.e. the number of videos.
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`.
            :math:`T_{in}/T_{out}` is a length of input/output sequence, i.e. the number of frames in a video.
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = SCN(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
