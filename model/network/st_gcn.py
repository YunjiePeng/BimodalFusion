import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .graph import SpatialGraph
from .scn import SCN
from .sync_batchnorm import SynchronizedBatchNorm2d

class ST_GCN(nn.Module):
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
        A = torch.tensor(self.graph.get_adjacency(semantic_level=0), dtype=torch.float32, requires_grad=False)

        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = self.graph.num_A
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        #---For CASIA-B---
        out_channel_list = [int(out_channels / 4), int(out_channels / 2)]
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, out_channel_list[0], kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(out_channel_list[0], out_channel_list[0], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[0], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channel_list[1], kernel_size, 1, **kwargs),
            st_gcn_block(out_channel_list[1], out_channels, kernel_size, 1, **kwargs),
            st_gcn_block(out_channels, out_channels, kernel_size, 1, **kwargs),
        ))
        
        #---For OUMVLP---
        # out_channel_list = [int(out_channels / 8), int(out_channels / 4), int(out_channels / 2)]
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn_block(in_channels, out_channel_list[0], kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn_block(out_channel_list[0], out_channel_list[0], kernel_size, 1, **kwargs),
        #     st_gcn_block(out_channel_list[0], out_channel_list[1], kernel_size, 1, **kwargs),
        #     st_gcn_block(out_channel_list[1], out_channel_list[1], kernel_size, 1, **kwargs),
        #     st_gcn_block(out_channel_list[1], out_channel_list[2], kernel_size, 1, **kwargs),
        #     st_gcn_block(out_channel_list[2], out_channel_list[2], kernel_size, 1, **kwargs),
        #     st_gcn_block(out_channel_list[2], out_channels, kernel_size, 1, **kwargs),
        #     st_gcn_block(out_channels, out_channels, kernel_size, 1, **kwargs),
        # ))
        
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

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

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, T*V, C)

        return x

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
            SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            SynchronizedBatchNorm2d(out_channels),
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
                SynchronizedBatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
