import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityLoss(nn.Module):
    def __init__(self, batch_size):
        super(SimilarityLoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, sil_feature, gray_feature):
        # sil_feature/gray_feature: [n, m, d]
        print("------in SimilarityLoss")
        print("sil_feature.size: ", sil_feature.size())
        print("gray_feature.size: ", gray_feature.size())
        print("\n")

        similarity_loss = self.batch_normloss(sil_feature, gray_feature)

        return similarity_loss

    def batch_normloss(self, x, y):
        norm_x = torch.norm(x, p=2, dim=2).unsqueeze(2)
        norm_y = torch.norm(y, p=2, dim=2).unsqueeze(2)
        norm_loss = torch.abs(norm_x - norm_y)
        print("norm_loss.size: ", norm_loss.size())
        return norm_loss
