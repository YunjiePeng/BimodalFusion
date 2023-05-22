import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPartCrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=False, eps=0.1):
        super(MultiPartCrossEntropyLoss, self).__init__()
        self.label_smooth = label_smooth
        self.eps = eps

    def forward(self, logits, labels):
        """
            logits: [p, n, c]
            labels: [n]
        """
        print("logits:{}, labels:{}".format(logits.size(), labels.size()))
        p, _, c = logits.size()
        log_preds = F.log_softmax(logits, dim=-1)  # [p, n, c]
        one_hot_labels = self.label2one_hot(
            labels, c).unsqueeze(0).repeat(p, 1, 1)  # [p, n, c]
        loss = self.compute_loss(log_preds, one_hot_labels)
        pred = logits.argmax(dim=-1)  # [p, n]
        accu = (pred == labels.unsqueeze(0)).float().mean()
        return loss, accu

    def compute_loss(self, predis, labels):
        softmax_loss = -(labels * predis).sum(-1)  # [p, n]
        losses = softmax_loss.mean(-1)

        if self.label_smooth:
            smooth_loss = - predis.mean(dim=-1)  # [p, n]
            smooth_loss = smooth_loss.mean()  # [p]
            smooth_loss = smooth_loss * self.eps
            losses = smooth_loss + losses * (1. - self.eps)
        return losses

    def label2one_hot(self, label, class_num):
        label = label.unsqueeze(-1)
        batch_size = label.size(0)
        device = label.device
        return torch.zeros(batch_size, class_num).to(device).scatter(1, label, 1)
