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

from .network import MSGG_2Layer
from .loss import TripletLoss
from .utils import DistributedTripletSampler, DistributedLossWrapper

class Model_MSGG2Layer:
    def __init__(self,
                 model_cfg,
                 lr,
                 margin,
                 frame_num,
                 batch_size,
                 num_workers,
                 restore_iter,
                 step_iter,
                 total_iter,
                 hard_or_full_trip,
                 model_name,
                 train_source,
                 test_source,
                 train_pid_num,
                 save_name,
                 state,
                 local_rank, ):
        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.lr = lr
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

        self.encoder = MSGG_2Layer(**model_cfg).float()
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        self.triplet_loss_x = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss_y = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()

        self.cross_entropy_loss_y = nn.CrossEntropyLoss().float()

        if state == 'train':
            dist.init_process_group(backend='nccl')

            self.device = torch.device('cuda:{}'.format(local_rank))
            self.encoder = self.encoder.to(self.device)
            self.encoder = nn.parallel.DistributedDataParallel(self.encoder, device_ids=[local_rank],
                                                               output_device=local_rank)

            self.triplet_loss_x = DistributedLossWrapper(self.triplet_loss_x)
            self.triplet_loss_x.cuda(self.device)

            self.triplet_loss_y = DistributedLossWrapper(self.triplet_loss_y)
            self.triplet_loss_y.cuda(self.device)

            self.cross_entropy_loss_y = DistributedLossWrapper(self.cross_entropy_loss_y)
            self.cross_entropy_loss_y.cuda(self.device)

        elif state == 'test':
            self.encoder.cuda()
            self.triplet_loss_x.cuda()
            self.triplet_loss_y.cuda()
            self.cross_entropy_loss_y.cuda()

        self.optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr,
                                   weight_decay=5e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_iter, gamma=0.1)

        self.hard_loss_metric_x = []
        self.full_loss_metric_x = []
        self.full_loss_num_x = []
        self.dist_list_x = []
        self.mean_dist_x = 0.01

        self.hard_loss_metric_y = []
        self.full_loss_metric_y = []
        self.full_loss_num_y = []
        self.dist_list_y = []
        self.mean_dist_y = 0.01

        self.cross_entropy_loss_metric_y = []

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
            if self.sample_type == 'random':  # for train
                # Random select consecutive frames
                random_range = len(frame_set) - self.frame_num + 1
                random_index = np.random.randint(random_range)
                frame_id_list = frame_set[random_index: random_index + self.frame_num]
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':  # for train
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:  # for test
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
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def clear_metric(self):
        self.hard_loss_metric_x = []
        self.full_loss_metric_x = []
        self.full_loss_num_x = []
        self.dist_list_x = []

        self.hard_loss_metric_y = []
        self.full_loss_metric_y = []
        self.full_loss_num_y = []
        self.dist_list_y = []

        self.cross_entropy_loss_metric_y = []

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        triplet_sampler = DistributedTripletSampler(dataset=self.train_source, batch_size=self.batch_size,
                                                    num_replicas=self.ngpu, rank=self.local_rank)
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

            feature_x, feature_y, feature_y_fc, y_cls_score = self.encoder(*seq, batch_frame)

            
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature_x = feature_x.permute(1, 0, 2).contiguous()
            triplet_label_x = target_label.unsqueeze(0).repeat(triplet_feature_x.size(0), 1)

            (full_loss_metric_x, hard_loss_metric_x, mean_dist_x, full_loss_num_x
             ) = self.triplet_loss_x(triplet_feature_x, triplet_label_x, dim=1)

            triplet_feature_y = feature_y_fc.permute(1, 0, 2).contiguous()
            triplet_label_y = target_label.unsqueeze(0).repeat(triplet_feature_y.size(0), 1)
            (full_loss_metric_y, hard_loss_metric_y, mean_dist_y, full_loss_num_y
             ) = self.triplet_loss_y(triplet_feature_y, triplet_label_y, dim=1)

            cross_entropy_loss_metric = self.cross_entropy_loss_y(y_cls_score, target_label, dim=0)
            
            alpha = 2.0
            beta = 1.0
            if self.hard_or_full_trip == 'hard':
                #loss = alpha * hard_loss_metric_x.mean() + beta * hard_loss_metric_y.mean() + gamma * hard_loss_metric_z.mean() + cross_entropy_loss_metric.mean()
                loss = alpha * hard_loss_metric_x.mean() + beta * hard_loss_metric_y.mean() + cross_entropy_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                #loss = alpha * full_loss_metric_x.mean() + beta * full_loss_metric_y.mean() + gamma * full_loss_metric_z.mean() + cross_entropy_loss_metric.mean()
                loss = alpha * full_loss_metric_x.mean() + beta * full_loss_metric_y.mean() + cross_entropy_loss_metric.mean()

            self.hard_loss_metric_x.append(hard_loss_metric_x.mean().data.cpu().numpy())
            self.full_loss_metric_x.append(full_loss_metric_x.mean().data.cpu().numpy())
            self.full_loss_num_x.append(full_loss_num_x.mean().data.cpu().numpy())
            self.dist_list_x.append(mean_dist_x.mean().data.cpu().numpy())

            self.hard_loss_metric_y.append(hard_loss_metric_y.mean().data.cpu().numpy())
            self.full_loss_metric_y.append(full_loss_metric_y.mean().data.cpu().numpy())
            self.full_loss_num_y.append(full_loss_num_y.mean().data.cpu().numpy())
            self.dist_list_y.append(mean_dist_y.mean().data.cpu().numpy())

            self.cross_entropy_loss_metric_y.append(cross_entropy_loss_metric.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if self.restore_iter % 5000 == 0:
                self.save()

            if self.restore_iter % 1000 == 0:
                logging.info(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 100 == 0:
                log_str = 'local_rank: {}'.format(self.local_rank)
                log_str = log_str + '\niter {}:'.format(self.restore_iter)
                log_str = log_str + ', x---hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric_x))
                log_str = log_str + ', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric_x))
                log_str = log_str + ', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num_x))
                self.mean_dist_x = np.mean(self.dist_list_x)
                log_str = log_str + ', mean_dist={0:.8f}\n'.format(self.mean_dist_x)

                log_str = log_str + 'iter {}:'.format(self.restore_iter)
                log_str = log_str + ', y---hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric_y))
                log_str = log_str + ', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric_y))
                log_str = log_str + ', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num_y))
                self.mean_dist_y = np.mean(self.dist_list_y)
                log_str = log_str + ', mean_dist={0:.8f}'.format(self.mean_dist_y)
                log_str = log_str + ', softmax_loss={0:.8f}'.format(np.mean(self.cross_entropy_loss_metric_y))

                log_str = log_str + ', lr=%f' % self.optimizer.param_groups[0]['lr']
                log_str = log_str + ', hard or full=%r\n' % self.hard_or_full_trip
                logging.info(log_str)

                sys.stdout.flush()
                self.clear_metric()

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
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            _, _, feature, cls_score = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
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
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
