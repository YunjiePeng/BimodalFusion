# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:00:58 2020

@author: watrix
"""

import torch
import diffdist
from torch.nn.parallel import DistributedDataParallel as DDP

# modified from https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/utils/distributed.py
def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

# modified from https://github.com/JohnGiorgi/DeCLUTR
def all_gather(embeddings, labels, dim=0):
    labels = labels.to(embeddings.device)
    # If we are not using distributed training, this is a no-op.
    if not is_distributed():
        return embeddings, labels
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    # Gather the embeddings on all replicas
    embeddings_list = [torch.ones_like(embeddings) for _ in range(world_size)]
    labels_list = [torch.ones_like(labels) for _ in range(world_size)]
#    torch.distributed.all_gather(embeddings_list, embeddings.contiguous())
#    torch.distributed.all_gather(labels_list, labels.contiguous())
    embeddings_list = diffdist.functional.all_gather(embeddings_list, embeddings.contiguous())
    labels_list = diffdist.functional.all_gather(labels_list, labels.contiguous())
    # The gathered copy of the current replicas embeddings have no gradients, so we overwrite
    # them with the embeddings generated on this replica, which DO have gradients.
#    embeddings_list[rank] = embeddings
#    labels_list[rank] = labels
    # Finally, we concatenate the embeddings
    embeddings = torch.cat(embeddings_list, dim=dim)
    labels = torch.cat(labels_list, dim=dim)
    return embeddings, labels

def gather_embeddings(embeddings, gpu_num, dim=0):
    # Gather the embeddings on all replicas
    embeddings_list = [torch.ones_like(embeddings) for _ in range(gpu_num)]
    embeddings_list = diffdist.functional.all_gather(embeddings_list, embeddings.contiguous())
    embeddings = torch.cat(embeddings_list, dim=dim)
    return embeddings

def all_gather_embeddings_labels(embeddings, labels, dim=0):
    if is_list_or_tuple(embeddings):
        assert is_list_or_tuple(labels)
        all_embeddings, all_labels = [], []
        for i in range(len(embeddings)):
            E, L = all_gather(embeddings[i], labels[i])
            all_embeddings.append(E)
            all_labels.append(L)
        embeddings = torch.cat(all_embeddings, dim=dim)
        labels = torch.cat(all_labels, dim=dim)
    else:
        embeddings, labels = all_gather(embeddings, labels, dim=dim)

    return embeddings, labels

class DistributedLossWrapper(torch.nn.Module):
    def __init__(self, loss, **kwargs):
        super().__init__()
        has_parameters = len([p for p in loss.parameters()]) > 0
        self.loss = DDP(loss, **kwargs) if has_parameters else loss

    def forward(self, embeddings, labels, dim, *args, **kwargs):
        embeddings, labels = all_gather_embeddings_labels(embeddings, labels, dim)
        return self.loss(embeddings, labels, *args, **kwargs)

class DistributedMinerWrapper(torch.nn.Module):
    def __init__(self, miner):
        super().__init__()
        self.miner = miner

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        embeddings, labels = all_gather_embeddings_labels(embeddings, labels)
        if ref_emb is not None:
            ref_emb, ref_labels = all_gather_embeddings_labels(ref_emb, ref_labels)
        return self.miner(embeddings, labels, ref_emb, ref_labels)
