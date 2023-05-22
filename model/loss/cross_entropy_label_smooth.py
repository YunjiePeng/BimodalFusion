import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        epsilon (float): weight.
    """
    def __init__(self, eps=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = eps
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        n, c = inputs.size()
        log_preds = self.logsoftmax(inputs) # [n, c]
        one_hot_labels = self.label2oneHot(targets, c) # [n, c]
        loss = self.compute_loss(log_preds, one_hot_labels)
        return loss
    
    def compute_loss(self, preds, labels):
        smooth_labels = (1 - self.epsilon) * labels + self.epsilon / preds.size()[1]
        loss = (- preds * smooth_labels).mean(0).sum()
        # logging.info("smooth_labels:{}, preds:{}, (- preds * smooth_labels):{}, loss:{}".format(smooth_labels.size(), preds.size(), (- preds * smooth_labels).size(), loss))
        return loss
    
    def label2oneHot(self, label, class_num):
        label = label.unsqueeze(-1)
        batch_size = label.size(0)
        device = label.device
        return torch.zeros(batch_size, class_num).to(device).scatter(1, label, 1)

# class CrossEntropyLabelSmooth(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.

#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.

#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """
#     def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
#         if self.use_gpu: targets = targets.cuda()
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = (- targets * log_probs).mean(0).sum()
#         return loss