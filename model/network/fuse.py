import torch
import torch.nn as nn
import numpy as np

class FuseNet(nn.Module):
    def __init__(self, sil_dim, ske_dim, part_num, out_channels):
        super(FuseNet, self).__init__()
        self.part_num = part_num
        
        # Reduce dim for ske.
        ske_reduce_dim = 32
        self.bn = nn.BatchNorm1d(ske_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.dp = nn.Dropout(p=0.3) # for CASIA-B
        self.dp = nn.Dropout(p=0.65) # for OUMVLP
        self.ske_reduce_fc = nn.Linear(ske_dim, ske_reduce_dim)

        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(self.part_num, sil_dim + ske_reduce_dim, out_channels)))])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)


    def forward(self, sil_feature, ske_feature):
        # print("sil_feature:", sil_feature.size(), ", ske_feature:", ske_feature.size())
        sil_feature = sil_feature.permute(1, 0, 2).contiguous() # [p, n, c]

        n, p, d = ske_feature.size()
        bn_ske_feature = self.bn(ske_feature.view(n, -1))
        relu_bn_ske_feature = self.relu(bn_ske_feature)
        dp_relu_bn_ske_feature = self.dp(relu_bn_ske_feature)
        ske_reduce_feature = self.ske_reduce_fc(dp_relu_bn_ske_feature).unsqueeze(0).contiguous()
        
        ske_feature = ske_reduce_feature.repeat(sil_feature.size()[0], 1, 1).contiguous()
        # print("in FuseNet, sil_feature:", sil_feature.size(), ", ske_feature:", ske_feature.size())
        fusion_feature = torch.cat([sil_feature, ske_feature], 2).contiguous()
        # print("fusion_feature:", fusion_feature.size())
        fusion_feature = fusion_feature.matmul(self.fc_bin[0])
        #print("fusion_feature after fc:", fusion_feature.size())

        #bn_fusion_feature = self.bn_neck(fusion_feature.view(n, -1))
        #cls_score = self.cls(bn_fusion_feature)
        #print("cls_score:", cls_score.size())

        fusion_feature = fusion_feature.permute(1, 0, 2).contiguous()
        # print("fusion_feature return:", fusion_feature.size())
        return fusion_feature
        #return fusion_feature, cls_score
