import sys
import math
import logging
import os
import os.path as osp
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
import torch.distributed as dist

from .network import GaitGL
from .loss import TripletLoss
from .utils import DistributedTripletSampler, DistributedLossWrapper, gather_embeddings

class Model_CNN_GaitGL:
    def __init__(self,
                 model_cfg,
                 lr,
                 margin,
                 frame_num,
                 batch_size,
                 num_workers,
                 restore_iter,
                 milestones,
                 total_iter,
                 hard_or_full_trip,
                 model_name,
                 train_source,
                 test_source,
                 train_pid_num,
                 save_name,
                 state,
                 local_rank,):
        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.lr = lr
        self.fuse_lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.state = state
        self.local_rank = local_rank
        self.ngpu = torch.cuda.device_count()

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.encoder = GaitGL(**model_cfg).float()
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.cross_entropy_loss = nn.CrossEntropyLoss().float()

        dist.init_process_group(backend='nccl')

        self.device = torch.device('cuda:{}'.format(local_rank))
        torch.cuda.set_device(local_rank)
        self.encoder.to(self.device)
        self.encoder = nn.parallel.DistributedDataParallel(self.encoder, device_ids=[local_rank],
                                                            output_device=local_rank)
        self.triplet_loss.to(self.device)
        self.triplet_loss = DistributedLossWrapper(self.triplet_loss)
        self.cross_entropy_loss.to(self.device)
        self.cross_entropy_loss = DistributedLossWrapper(self.cross_entropy_loss)

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr, weight_decay=5.0e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.cross_entropy_loss_metric = []

        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                seq_len = len(frame_set)
                frame_id_list = list(range(seq_len))
                if seq_len < self.frame_num:
                    it = math.ceil(self.frame_num / seq_len)
                    seq_len = seq_len * it
                    frame_id_list = frame_id_list * it

                start = np.random.choice(list(range(0, seq_len - self.frame_num + 1)))
                end = start + self.frame_num
                _frame_id_list = list(range(seq_len))
                _frame_id_list = _frame_id_list[start:end]
                _frame_id_list = sorted(np.random.choice(
                    _frame_id_list, self.frame_num, replace=False))
                frame_id_list = [frame_id_list[i] for i in _frame_id_list]

                # if len(frame_set) < self.frame_num:
                #     print("len(frame_set):{}, frame_id_list:{}".format(len(frame_set), frame_id_list))

                #---FrameMin30, Delta0---
                # random_range = len(frame_set) - self.frame_num + 1
                # random_index = np.random.randint(random_range)
                # frame_id_list = frame_set[random_index: random_index + self.frame_num]

                #---FrameMin15, Delta10---
                # if len(frame_set) >= self.frame_num:
                #     start = np.random.choice(list(range(len(frame_set)-self.frame_num+1)), 1, replace=False)
                #     start = int(start)
                #     frame_id_list = frame_set[start:start+self.frame_num+10] # 5 or 10
                #     frame_id_list = sorted(np.random.choice(frame_id_list, self.frame_num, replace=False))
                # else:
                #     frame_id_list = sorted(np.random.choice(frame_set, self.frame_num, replace=True))

                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)

            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]

            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)

            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])

            seqs = [[np.concatenate([
                            seqs[i][j]
                            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                            if i < batch_size], 0)
                        for _ in range(gpu_num)]
                    for j in range(feature_num)]

            seqs = [np.asarray([
                            np.pad(seqs[j][_],
                            ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)), 'constant',constant_values=0)
                        for _ in range(gpu_num)])
                    for j in range(feature_num)]

            if self.local_rank < batch_size:
                index_start = self.local_rank * batch_per_gpu
                index_end = index_start + batch_per_gpu
            else:
                index_start = 0
                index_end = index_start + batch_per_gpu

            seqs = [seq[index_start:index_end] for seq in seqs]
            batch_frames = batch_frames[index_start:index_end]

            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = DistributedTripletSampler(self.train_source, self.batch_size, batch_shuffle=True)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, cls_score = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label, dim=1)

            # print("cls_score:{}, target_lable:{}".format(cls_score.size(), target_label.size()))
            p, n, c = cls_score.size()
            cls_score = cls_score.view(-1, c)
            target_label = target_label.repeat(p)
            cross_entropy_loss_metric = self.cross_entropy_loss(cls_score, target_label, dim=0)

            # cross_entropy_loss_metric, cross_entropy_accuracy_metric = self.cross_entropy_loss(cls_score, target_label, dim=0)
            # cross_entropy_loss_metric = cross_entropy_loss_metric.mean(-1)

            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean() + cross_entropy_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean() + cross_entropy_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            self.cross_entropy_loss_metric.append(cross_entropy_loss_metric.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if self.restore_iter % 2000 == 0:
                self.save()
                logging.info(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 100 == 0:
                log_str = 'iter {}:'.format(self.restore_iter)
                log_str = log_str + ', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric))
                log_str = log_str + ', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric))
                log_str = log_str + ', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num))
                self.mean_dist = np.mean(self.dist_list)
                log_str = log_str + ', mean_dist={0:.8f}'.format(self.mean_dist)
                log_str = log_str + ', lr=%f' % self.optimizer.param_groups[0]['lr']
                log_str = log_str + ', hard or full=%r' % self.hard_or_full_trip
                log_str = log_str + ', softmax_loss={0:.8f}'.format(np.mean(self.cross_entropy_loss_metric))
                logging.info(log_str)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []
                self.cross_entropy_loss_metric = []
                self.cross_entropy_accuracy_metric = []

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            # logging.info("label:{}, seq_type:{}, view:{}".format(label, seq_type, view))
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, _ = self.encoder(*seq, batch_frame)
            gather_feature = gather_embeddings(feature, torch.cuda.device_count())[:len(label)]
            n, num_bin, _ = gather_feature.size()
            feature_list.append(gather_feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        if self.local_rank == 0:
            torch.save(self.encoder.module.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-encoder.ptm'.format(
                                    self.save_name, self.restore_iter)))
            torch.save(self.optimizer.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-optimizer.ptm'.format(
                                    self.save_name, self.restore_iter)))

        # restore_iter: iteration index of the checkpoint to load

    def load(self, restore_iter):
        self.encoder.module.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        # self.optimizer.load_state_dict(torch.load(osp.join(
        #     'checkpoint', self.model_name,
        #     '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))

        # ckp_prefix: prefix of the checkpoint to load

    def load_ckp(self, ckp_prefix):
        self.encoder.load_state_dict(torch.load('../{}encoder.ptm'.format(ckp_prefix)))
        self.optimizer.load_state_dict(torch.load('../{}optimizer.ptm'.format(ckp_prefix)))

