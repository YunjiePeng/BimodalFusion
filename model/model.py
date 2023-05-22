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
import torch.cuda.amp as amp

from .network import GaitPart, GaitGL, MSGG, FuseNet
from .loss import TripletLoss
from .utils import DistributedTripletSampler, DistributedLossWrapper, gather_embeddings

class Model:
    def __init__(self,
                 pretrain_gcn,
                 pretrain_cnn,
                 cnn_branch,
                 cnn_cfg,
                 pgcn_cfg,
                 fuse_cfg,
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
                 local_rank,):
        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source
        self.cnn_branch = cnn_branch
        self.pretrain_cnn = pretrain_cnn
        self.pretrain_gcn = pretrain_gcn

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

        self.pgcn_cfg = pgcn_cfg
        self.cnn_cfg = cnn_cfg
        self.fuse_cfg = fuse_cfg

        self.gcn_encoder = MSGG(**pgcn_cfg).float()
        self.gcn_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.gcn_encoder)
        self.gcn_tripletloss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.gcn_crossEntropyloss = nn.CrossEntropyLoss().float()

        self.enable_amp = False

        if cnn_branch == 'gaitpart':
            self.cnn_encoder = GaitPart(**cnn_cfg).float()
            self.cnn_gcn_lr = 1e-4
        elif cnn_branch == 'gaitgl':
            self.cnn_encoder = GaitGL(**cnn_cfg).float()
            self.cnn_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.cnn_encoder)
            self.cnn_crossEntropyloss = nn.CrossEntropyLoss().float()
            self.cnn_gcn_lr = 1e-4
            self.enbale_amp = True
            self.scalar = amp.GradScaler()

        self.cnn_tripletloss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()

        self.fuse_encoder = FuseNet(sil_dim=cnn_cfg['hidden_dim'], ske_dim=pgcn_cfg['out_channels'], **fuse_cfg)
        self.fuse_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.fuse_encoder)
        self.fuse_tripletloss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()

        dist.init_process_group(backend='nccl')

        self.device = torch.device('cuda:{}'.format(local_rank))

        self.gcn_encoder.to(self.device)
        self.gcn_encoder = nn.parallel.DistributedDataParallel(self.gcn_encoder, device_ids=[local_rank], output_device=local_rank)
        self.gcn_tripletloss.to(self.device)
        self.gcn_tripletloss = DistributedLossWrapper(self.gcn_tripletloss)
        self.gcn_crossEntropyloss.to(self.device)
        self.gcn_crossEntropyloss = DistributedLossWrapper(self.gcn_crossEntropyloss)

        self.cnn_encoder.to(self.device)
        self.cnn_encoder = nn.parallel.DistributedDataParallel(self.cnn_encoder, device_ids=[local_rank], output_device=local_rank)
        self.cnn_tripletloss.to(self.device)
        self.cnn_tripletloss = DistributedLossWrapper(self.cnn_tripletloss)
        if self.cnn_branch == 'gaitgl':
            self.cnn_crossEntropyloss.to(self.device)
            self.cnn_crossEntropyloss = DistributedLossWrapper(self.cnn_crossEntropyloss)
        
        self.fuse_encoder.to(self.device)
        self.fuse_encoder = nn.parallel.DistributedDataParallel(self.fuse_encoder, device_ids=[local_rank], output_device=local_rank)
        self.fuse_tripletloss.to(self.device)
        self.fuse_tripletloss = DistributedLossWrapper(self.fuse_tripletloss)

        self.fuse_optimizer = optim.SGD([
            {'params': self.fuse_encoder.parameters()},
        ], lr=self.fuse_lr, weight_decay=5e-4, momentum=0.9)
        self.fuse_scheduler = optim.lr_scheduler.StepLR(self.fuse_optimizer, step_size=step_iter, gamma=0.1)

        self.branch_optimizer = optim.SGD([
            {'params': self.gcn_encoder.parameters()},
            {'params': self.cnn_encoder.parameters()},
        ], lr=self.cnn_gcn_lr, weight_decay=5e-4, momentum=0.9)
        self.branch_scheduler = optim.lr_scheduler.StepLR(self.branch_optimizer, step_size=step_iter, gamma=0.1)
        
        self.gcn_hard_loss_metric = []
        self.gcn_full_loss_metric = []
        self.gcn_full_loss_num = []
        self.gcn_dist_list = []
        self.gcn_mean_dist = 0.01
        self.gcn_crossEntropy_loss_metric = []
        
        self.cnn_hard_loss_metric = []
        self.cnn_full_loss_metric = []
        self.cnn_full_loss_num = []
        self.cnn_dist_list = []
        self.cnn_mean_dist = 0.01
        if self.cnn_branch == 'gaitgl':
            self.cnn_crossEntropy_loss_metric = []

        self.fuse_hard_loss_metric = []
        self.fuse_full_loss_metric = []
        self.fuse_full_loss_num = []
        self.fuse_dist_list = []
        self.fuse_mean_dist = 0.01

        self.sample_type = 'all'

    def collate_fn(self, batch):
        ''' batch[x][y][z]
            x: the number of sequence.
            y: 0(data), 1(frame_set), 2(view), 3(seq_type), 4(label)
            z: 0(silhouette data), 1(skeleton data)
        '''
        batch_size = len(batch)
        feature_num = len(batch[0][0][0])

        sil_seqs = [batch[i][0][0] for i in range(batch_size)]
        sil_frame_sets = [batch[i][1][0] for i in range(batch_size)]
        keypoints_seqs = [batch[i][0][1] for i in range(batch_size)]
        keypoints_frame_sets = [batch[i][1][1] for i in range(batch_size)]

        seqs = [sil_seqs, keypoints_seqs]
        frame_sets = [sil_frame_sets, keypoints_frame_sets]

        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        # logging.info("local_rank:{}, label:{}, seq_type:{}, view:{}\n".format(self.local_rank, label, seq_type, view))
        def select_frame(index):
            sil_sample = seqs[0][index]
            keypoints_sample = seqs[1][index]
            sil_frame_set = frame_sets[0][index]
            ske_frame_set = frame_sets[1][index]
            if self.sample_type == 'random':
                if self.cnn_branch == 'gaitpart':
                    # Random10, the same as the training process in GaitPart
                    if len(sil_frame_set) >= self.frame_num['silhouette']:
                        sil_random_range = len(sil_frame_set) - self.frame_num['silhouette'] + 1
                        sil_random_index = np.random.randint(sil_random_range)
                        sil_frame_id_list = sil_frame_set[sil_random_index:sil_random_index + self.frame_num['silhouette'] + 10]
                        sil_frame_id_list = sorted(np.random.choice(sil_frame_id_list, self.frame_num['silhouette'], replace=False))
                    else:
                        sil_frame_id_list = sorted(np.random.choice(sil_frame_set, self.frame_num['silhouette'], replace=True))
                elif self.cnn_branch == 'gaitgl':
                    # Continues30, the same as the training process in GaitGL
                    sil_seq_len = len(sil_frame_set)
                    sil_frame_id_list = list(range(sil_seq_len))
                    if sil_seq_len < self.frame_num['silhouette']:
                        it = math.ceil(self.frame_num['silhouette'] / sil_seq_len)
                        sil_seq_len = sil_seq_len * it
                        sil_frame_id_list = sil_frame_id_list * it

                    start = np.random.choice(list(range(0, sil_seq_len - self.frame_num['silhouette'] + 1)))
                    end = start + self.frame_num['silhouette']
                    _sil_frame_id_list = list(range(sil_seq_len))
                    _sil_frame_id_list = _sil_frame_id_list[start:end]
                    _sil_frame_id_list = sorted(np.random.choice(
                                            _sil_frame_id_list, self.frame_num['silhouette'], replace=False))
                    sil_frame_id_list = [sil_frame_id_list[i] for i in _sil_frame_id_list]

                ske_random_range = len(ske_frame_set) - self.frame_num['skeleton'] + 1
                ske_random_index = np.random.randint(ske_random_range)
                ske_frame_id_list = ske_frame_set[ske_random_index:ske_random_index + self.frame_num['skeleton']]

                _sil_seqs = [feature.loc[sil_frame_id_list].values for feature in sil_sample]
                _keypoints_seqs = [feature.loc[ske_frame_id_list].values for feature in keypoints_sample]
                _ = [_sil_seqs, _keypoints_seqs]
            else:
                _sil_seqs = [feature.values for feature in sil_sample]
                _keypoints_seqs = [feature.values for feature in keypoints_sample]
                _ = [_sil_seqs, _keypoints_seqs]
            return _

        '''
        seqs[x][y][z]:
        x-第几个seq
        y-0(sil_seqs) 1(gray_seqs)
        z-第几张图片
        '''
        seqs = list(map(select_frame, range(len(seqs[0]))))
        # print("len(seqs):{}, seqs[0][0].shape:{}".format(len(seqs), seqs[0][0].shape))

        if self.sample_type == 'random':
            sil_seqs = [np.asarray([seqs[i][0][j] for i in range(batch_size)]) for j in range(feature_num)]
            keypoints_seqs = [np.asarray([seqs[i][1][j] for i in range(batch_size)]) for j in range(feature_num)]
            seqs = [sil_seqs, keypoints_seqs]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)

            #---Silhouettes---
            sil_batch_frames = [[
                len(sil_frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]

            if len(sil_batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(sil_batch_frames[-1])):
                    sil_batch_frames[-1].append(0)

            sil_max_sum_frame = np.max([np.sum(sil_batch_frames[_]) for _ in range(gpu_num)])
            sil_seqs = [[np.concatenate([seqs[i][0][j]
                                         for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1)) if i < batch_size],
                                        0)
                         for _ in range(gpu_num)]
                        for j in range(feature_num)]

            sil_seqs = [np.asarray([np.pad(sil_seqs[j][_],
                                           ((0, sil_max_sum_frame - np.array(sil_seqs[j][_]).shape[0]), (0, 0), (0, 0)),
                                           'constant', constant_values=0)
                                    for _ in range(gpu_num)])
                        for j in range(feature_num)]

            #---Skeletons---
            keypoints_batch_frames = [[
                len(keypoints_frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            
            if len(keypoints_batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(keypoints_batch_frames[-1])):
                    keypoints_batch_frames[-1].append(0)

            keypoints_max_sum_frame = np.max([np.sum(keypoints_batch_frames[_]) for _ in range(gpu_num)])
            keypoints_seqs = [[np.concatenate([seqs[i][1][j]
                                               for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1)) if
                                               i < batch_size], 0)
                               for _ in range(gpu_num)]
                              for j in range(feature_num)]

            keypoints_seqs = [np.asarray([np.pad(keypoints_seqs[j][_],
                                                 ((0, keypoints_max_sum_frame - np.array(keypoints_seqs[j][_]).shape[0]), (0, 0), (0, 0)),
                                                 'constant', constant_values=0)
                                          for _ in range(gpu_num)])
                              for j in range(feature_num)]

            if self.local_rank < batch_size:
                index_start = self.local_rank * batch_per_gpu
                index_end = index_start + batch_per_gpu
            else:
                index_start = 0
                index_end = index_start + batch_per_gpu
            
            cur_gpu_sil_seqs = [sil_seq[index_start:index_end] for sil_seq in sil_seqs]
            cur_gpu_keypoints_seqs = [keypoints_seq[index_start:index_end] for keypoints_seq in keypoints_seqs]
            cur_gpu_batch_frames = [sil_batch_frames[index_start:index_end], keypoints_batch_frames[index_start:index_end]]
            # print("local_rank:{}, index_start:{}, index_end:{}, batch_frames:{}, cur:{}, batch_size:{}".format(self.local_rank, index_start, index_end, batch_frames, cur_gpu_batch_frames, batch_size))
            
            seqs = [cur_gpu_sil_seqs, cur_gpu_keypoints_seqs]
            batch[4] = np.asarray(cur_gpu_batch_frames)

        batch[0] = seqs
        return batch

    def metric_clear(self):
        self.gcn_hard_loss_metric = []
        self.gcn_fpull_loss_metric = []
        self.gcn_full_loss_num = []
        self.gcn_dist_list = []
        self.gcn_crossEntropy_loss_metric = []
        #self.gcn_hard_loss_metric_x = []
        #self.gcn_full_loss_metric_x = []
        #self.gcn_full_loss_num_x = []
        #self.gcn_dist_list_x = []

        #self.gcn_hard_loss_metric_y = []
        #self.gcn_full_loss_metric_y = []
        #self.gcn_full_loss_num_y = []
        #self.gcn_dist_list_y = []

        #self.gcn_hard_loss_metric_z = []
        #self.gcn_full_loss_metric_z = []
        #self.gcn_full_loss_num_z = []
        #self.gcn_dist_list_z = []
        
        self.cnn_hard_loss_metric = []
        self.cnn_full_loss_metric = []
        self.cnn_full_loss_num = []
        self.cnn_dist_list = []
        if self.cnn_branch == 'gaitgl':
            self.cnn_crossEntropy_loss_metric = []

        self.fuse_hard_loss_metric = []
        self.fuse_full_loss_metric = []
        self.fuse_full_loss_num = []
        self.fuse_dist_list = []
        #self.fuse_crossEntropy_loss_metric = []

    def forward_func(self, sil_seq, keypoints_seq, batch_frame, target_label):
        midlayer1_feature, midlayer2_feature, gcn_feature, gcn_feature_fc, gcn_cls_score = self.gcn_encoder(*keypoints_seq, batch_frame)
        del midlayer1_feature
        del midlayer2_feature
        gcn_triplet_feature = gcn_feature_fc.permute(1, 0, 2).contiguous()
        gcn_triplet_label = target_label.unsqueeze(0).repeat(gcn_triplet_feature.size(0), 1)
        (gcn_full_loss_metric, gcn_hard_loss_metric, gcn_mean_dist, gcn_full_loss_num
        ) = self.gcn_tripletloss(gcn_triplet_feature, gcn_triplet_label, dim=1)
        gcn_crossEntropyloss_metric = self.gcn_crossEntropyloss(gcn_cls_score, target_label, dim=0)

        if self.cnn_branch == 'gaitgl':
            cnn_feature_fc, cnn_cls_score = self.cnn_encoder(*sil_seq, batch_frame)
            fuse_feature = self.fuse_encoder(cnn_feature_fc, gcn_feature)

            p, n, c = cnn_cls_score.size()
            cnn_cls_score = cnn_cls_score.view(-1, c)
            cnn_target_label = target_label.repeat(p)
            cnn_crossEntropyloss_metric = self.cnn_crossEntropyloss(cnn_cls_score, cnn_target_label, dim=0)
        else:
            cnn_feature, cnn_feature_fc = self.cnn_encoder(*sil_seq, batch_frame)
            fuse_feature = self.fuse_encoder(cnn_feature, gcn_feature)

        cnn_triplet_feature = cnn_feature_fc.permute(1, 0, 2).contiguous()
        cnn_label = target_label.unsqueeze(0).repeat(cnn_triplet_feature.size(0), 1)
        (cnn_full_loss_metric, cnn_hard_loss_metric, cnn_mean_dist, cnn_full_loss_num
        ) = self.cnn_tripletloss(cnn_triplet_feature, cnn_label, dim=1)

        fuse_triplet_feature = fuse_feature.permute(1, 0, 2).contiguous()
        fuse_triplet_label = target_label.unsqueeze(0).repeat(fuse_triplet_feature.size(0), 1)
        (fuse_full_loss_metric, fuse_hard_loss_metric, fuse_mean_dist, fuse_full_loss_num
        ) = self.fuse_tripletloss(fuse_triplet_feature, fuse_triplet_label, dim=1)

        if self.hard_or_full_trip == 'hard':
            loss = gcn_crossEntropyloss_metric.mean() + gcn_hard_loss_metric.mean() + cnn_hard_loss_metric.mean() + fuse_hard_loss_metric.mean()# + fuse_crossEntropyloss_metric.mean()
        elif self.hard_or_full_trip == 'full':
            loss = gcn_crossEntropyloss_metric.mean() + gcn_full_loss_metric.mean() + cnn_full_loss_metric.mean() + fuse_full_loss_metric.mean()# + fuse_crossEntropyloss_metric.mean()

        if self.cnn_branch == 'gaitgl':
            loss = loss + cnn_crossEntropyloss_metric.mean()
            self.cnn_crossEntropy_loss_metric.append(cnn_crossEntropyloss_metric.mean().data.cpu().numpy())
        
        self.gcn_hard_loss_metric.append(gcn_hard_loss_metric.mean().data.cpu().numpy())
        self.gcn_full_loss_metric.append(gcn_full_loss_metric.mean().data.cpu().numpy())
        self.gcn_full_loss_num.append(gcn_full_loss_num.mean().data.cpu().numpy())
        self.gcn_dist_list.append(gcn_mean_dist.mean().data.cpu().numpy())
        self.gcn_crossEntropy_loss_metric.append(gcn_crossEntropyloss_metric.mean().data.cpu().numpy())
        
        self.cnn_hard_loss_metric.append(cnn_hard_loss_metric.mean().data.cpu().numpy())
        self.cnn_full_loss_metric.append(cnn_full_loss_metric.mean().data.cpu().numpy())
        self.cnn_full_loss_num.append(cnn_full_loss_num.mean().data.cpu().numpy())
        self.cnn_dist_list.append(cnn_mean_dist.mean().data.cpu().numpy())

        self.fuse_hard_loss_metric.append(fuse_hard_loss_metric.mean().data.cpu().numpy())
        self.fuse_full_loss_metric.append(fuse_full_loss_metric.mean().data.cpu().numpy())
        self.fuse_full_loss_num.append(fuse_full_loss_num.mean().data.cpu().numpy())
        self.fuse_dist_list.append(fuse_mean_dist.mean().data.cpu().numpy())

        return loss
        
    def fit(self):
        if self.restore_iter == 0:
            self.load_pretrain()
        else:
            self.load(self.restore_iter)

        self.gcn_encoder.train()
        self.cnn_encoder.train()
        self.fuse_encoder.train()
        self.sample_type = 'random'

        for param_group in self.fuse_optimizer.param_groups:
            param_group['lr'] = self.lr

        triplet_sampler = DistributedTripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            #print("---------\nlabel:", label, "\nseq_type:", seq_type, "\nview:", view)
            self.restore_iter += 1
            self.fuse_optimizer.zero_grad()
            self.branch_optimizer.zero_grad()

            sil_seq = seq[0]
            keypoints_seq = seq[1]

            for i in range(len(sil_seq)):
                sil_seq[i] = self.np2var(sil_seq[i]).float()
                keypoints_seq[i] = self.np2var(keypoints_seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            del seq

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            if self.enbale_amp:
                with amp.autocast():
                    loss = self.forward_func(sil_seq, keypoints_seq, batch_frame, target_label)

                if loss > 1e-9:
                    self.scalar.scale(loss).backward()
                    self.scalar.step(self.branch_optimizer)
                    self.scalar.step(self.branch_scheduler)
                    self.scalar.step(self.fuse_optimizer)
                    self.scalar.step(self.fuse_scheduler)
                    self.scalar.update()
            else:
                loss = self.forward_func(sil_seq, keypoints_seq, batch_frame, target_label)
                if loss > 1e-9:
                    loss.backward()
                    self.fuse_optimizer.step()
                    self.fuse_scheduler.step()
                    self.branch_optimizer.step()
                    self.branch_scheduler.step()

            if self.restore_iter % 1000 == 0:
                self.save()
                logging.info(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 100 == 0:
                log_str = '\nske {}:'.format(self.restore_iter)
                log_str = log_str + '     gcn-hard_loss_metric={0:.8f}'.format(np.mean(self.gcn_hard_loss_metric))
                log_str = log_str + ', full_loss_metric={0:.8f}'.format(np.mean(self.gcn_full_loss_metric))
                log_str = log_str + ', full_loss_num={0:.8f}'.format(np.mean(self.gcn_full_loss_num))
                self.gcn_mean_dist = np.mean(self.gcn_dist_list)
                log_str = log_str + ', mean_dist={0:.8f}'.format(self.gcn_mean_dist)
                log_str = log_str + ', softmax_loss={0:.8f}\n'.format(np.mean(self.gcn_crossEntropy_loss_metric))
                
                log_str = log_str + 'sil {}:'.format(self.restore_iter)
                log_str = log_str + '     cnn-hard_loss_metric={0:.8f}'.format(np.mean(self.cnn_hard_loss_metric))
                log_str = log_str + ', full_loss_metric={0:.8f}'.format(np.mean(self.cnn_full_loss_metric))
                log_str = log_str + ', full_loss_num={0:.8f}'.format(np.mean(self.cnn_full_loss_num))
                self.cnn_mean_dist = np.mean(self.cnn_dist_list)
                # log_str = log_str + ', mean_dist={0:.8f}\n'.format(self.cnn_mean_dist)
                if self.cnn_branch == 'gaitgl':
                    log_str = log_str + ', mean_dist={0:.8f}'.format(self.cnn_mean_dist)
                    log_str = log_str + ', softmax_loss={0:.8f}\n'.format(np.mean(self.cnn_crossEntropy_loss_metric))
                else:
                    log_str = log_str + ', mean_dist={0:.8f}\n'.format(self.cnn_mean_dist)

                log_str = log_str + 'fus {}:'.format(self.restore_iter)
                log_str = log_str + '    fuse-hard_loss_metric={0:.8f}'.format(np.mean(self.fuse_hard_loss_metric))
                log_str = log_str + ', full_loss_metric={0:.8f}'.format(np.mean(self.fuse_full_loss_metric))
                log_str = log_str + ', full_loss_num={0:.8f}'.format(np.mean(self.fuse_full_loss_num))
                self.fuse_mean_dist = np.mean(self.fuse_dist_list)
                log_str = log_str + ', mean_dist={0:.8f}'.format(self.fuse_mean_dist)
                #log_str = log_str + ', softmax_loss={0:.8f}'.format(np.mean(self.fuse_crossEntropy_loss_metric))
                log_str = log_str + ', lr=%f' % self.fuse_optimizer.param_groups[0]['lr']
                log_str = log_str + ', hard or full=%r\n' % self.hard_or_full_trip

                logging.info(log_str)
                sys.stdout.flush()
                self.metric_clear()

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        # self.load_pretrain_for_test()
        self.gcn_encoder.eval()
        self.cnn_encoder.eval()
        self.fuse_encoder.eval()

        #print("self.fuse_encoder.state_dict:", self.fuse_encoder.state_dict)
        #print("self.fuse_encoder.state_dict():", self.fuse_encoder.state_dict()['fc_bin.0'])
        #fuse_weight_matrix = self.fuse_encoder.state_dict()['fc_bin.0'].data.cpu().numpy()
        #np.save('./fuse_weight_matrix.npy', fuse_weight_matrix)
        
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
            # print("self.local_rank:{}, batch_frame:{}".format(self.local_rank, batch_frame))
            cnn_seq = seq[0]
            gcn_seq = seq[1]

            for j in range(len(cnn_seq)):
                gcn_seq[j] = self.np2var(gcn_seq[j]).float()
                cnn_seq[j] = self.np2var(cnn_seq[j]).float()

            if batch_frame is not None:
                gcn_batch_frame = self.np2var(batch_frame[1]).int()
                cnn_batch_frame = self.np2var(batch_frame[0]).int()

            _, _, gcn_feature, gcn_feature_fc, cls_score = self.gcn_encoder(*gcn_seq, gcn_batch_frame)
            if self.cnn_branch == 'gaitgl':
                cnn_feature_fc, _ = self.cnn_encoder(*cnn_seq, cnn_batch_frame)
                fusion_feature = self.fuse_encoder(cnn_feature_fc, gcn_feature)
            else:
                cnn_feature, cnn_feature_fc = self.cnn_encoder(*cnn_seq, cnn_batch_frame)
                fusion_feature = self.fuse_encoder(cnn_feature, gcn_feature)
            # print("local_rank:{}, fusion_feature:{}".format(self.local_rank, fusion_feature.size()))

            gather_feature = gather_embeddings(fusion_feature, torch.cuda.device_count())[:len(label)]
            # gather_feature = gather_embeddings(cnn_feature_fc, torch.cuda.device_count())[:len(label)]
            # gather_feature = gather_embeddings(gcn_feature_fc, torch.cuda.device_count())[:len(label)]

            # print("gather_feature:{}, gather_feature2:{}".format(gather_feature.size(), gather_feature2.size()))
            # print("local_rank:{}, fusion_feature:{}, gather_feature:{}".format(self.local_rank, fusion_feature.size(), gather_feature.size()))
            # print("gcn_feature_fc:", gcn_feature_fc.size(), ", cnn_feature_fc:", cnn_feature_fc.size(), ", fusion_feature:", fusion_feature.size())
            feature = gather_feature
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        if self.local_rank == 0:
            os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
            torch.save(self.gcn_encoder.module.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-gcn_encoder.ptm'.format(
                                    self.save_name, self.restore_iter)))
            torch.save(self.cnn_encoder.module.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-cnn_encoder.ptm'.format(
                                    self.save_name, self.restore_iter)))
            torch.save(self.fuse_encoder.module.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-fuse_encoder.ptm'.format(
                                    self.save_name, self.restore_iter)))
            torch.save(self.fuse_optimizer.state_dict(),
                       osp.join('checkpoint', self.model_name,
                                '{}-{:0>5}-optimizer.ptm'.format(
                                    self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.gcn_encoder.module.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-gcn_encoder.ptm'.format(self.save_name, restore_iter))))
        self.cnn_encoder.module.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-cnn_encoder.ptm'.format(self.save_name, restore_iter))))
        self.fuse_encoder.module.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-fuse_encoder.ptm'.format(self.save_name, restore_iter))))

    def load_pretrain(self):
        self.gcn_encoder.module.load_state_dict(torch.load(osp.join('checkpoint', self.pretrain_gcn)))
        self.cnn_encoder.module.load_state_dict(torch.load(osp.join('checkpoint', self.pretrain_cnn)))

    def load_pretrain_for_test(self):
        self.gcn_encoder.module.load_state_dict(torch.load(osp.join('checkpoint', self.pretrain_gcn)))
        self.cnn_encoder.module.load_state_dict(torch.load(osp.join('checkpoint', self.pretrain_cnn)))
