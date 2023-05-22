import math
import random

import torch
import torch.utils.data as tordata
import torch.distributed as dist

import logging

import numpy as np

class DistributedTripletSampler(tordata.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    Arguments:
        dataset: Dataset used for sampling.
        batch_size: A tuple contains the number of persons and the number of sequences for each person in a batch
        num_replicas (optional): Number of processes participating in distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, batch_size, batch_shuffle=False, num_replicas=None, rank=None):
        random.seed(2020)
        # random.seed(1234)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle
        self.num_replicas = num_replicas
        self.rank = rank

        assert self.batch_size[0] % self.num_replicas == 0

        self.num_samples_per_replica = int(self.batch_size[0] / self.num_replicas)
        self.label_list = sorted(list(self.dataset.label_set))

    def __iter__(self):
        while (True):
            pid_list = random.sample(self.label_list, self.batch_size[0])  # Random choose person ids.
            sampled_pid = pid_list[self.rank: self.batch_size[0]:self.num_replicas]  # for each replica
            sampled_indices = []

            for _id in sampled_pid:
                _index = self.dataset.index_dict.loc[_id, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random.choices(
                    _index,
                    k=self.batch_size[1])  # Random choose gait sequence.
                sampled_indices += _index
            # logging.info("sampled_indices:{}".format(sampled_indices))
            if self.batch_shuffle:
                sampled_indices = sync_random_sample_list(
                    sampled_indices, len(sampled_indices))

            yield sampled_indices

    def __len__(self):
        return self.num_samples_per_replica

def sync_random_sample_list(obj_list, k):
    idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]