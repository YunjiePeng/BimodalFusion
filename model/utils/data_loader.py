import os
import os.path as osp

import numpy as np
import pickle
import logging

from torchvision import transforms

from .data_set import DataSet
from .augmentation import JointNoise
from warnings import warn


def load_data(dataset, data_type, frame_num, resolution, pid_num, pid_shuffle, data_path, cache=True):
    logging.info("dataset:{}, data_type:{}, data_path:{}, cache:{}".format(dataset, data_type, data_path, cache))

    pid_fname = osp.join('partition', '{}_{}_{}_{}.npy'.format(dataset, data_type, pid_num, pid_shuffle))
    pid_list = []

    if osp.exists(pid_fname):
        pid_list = np.load(pid_fname, allow_pickle=True)
        seq_dir, view, seq_type, label = data_loading(frame_num, pid_num, data_type, dataset.split("_")[0], data_path, pid_list)
    else:
        seq_dir, view, seq_type, label = data_loading(frame_num, pid_num, data_type, dataset.split("_")[0], data_path, pid_list)
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)

        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)

    train_list = pid_list[0]
    test_list = pid_list[1]
    logging.info("pid_train:{}\npid_test:{}".format(train_list, test_list))
    logging.info("len(pid_train):{}, len(pid_test):{}".format(len(train_list), len(test_list)))

    ske_augment = None
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        data_type, cache, resolution, ske_augment)
    
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        data_type, cache, resolution)

    return train_source, test_source

def data_loading(frame_num, pid_num, data_type, dataset, data_path, pid_list):
    if dataset == 'CASIA-B':
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        include_005 = False
    else:
        include_005 = True

    if len(data_path) > 1:
        return load_all_data(frame_num, pid_num, data_type, dataset, data_path, include_005, pid_list)
    else:
        return load_single_data(frame_num, pid_num, data_type, dataset, data_path, include_005, pid_list)

def load_single_data(frame_num, pid_num, data_type, dataset, data_path, include_005, pid_list):
    if len(pid_list) == 2:
        train_label = pid_list[0]
        test_label = pid_list[1]

    data_path = data_path[0]
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    for _label in sorted(list(os.listdir(data_path))):
        if (include_005 is False) and (_label == '005'):
            continue

        label_path = osp.join(data_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                seq_info = [_label, _seq_type, _view]

                _seq_dir = osp.join(seq_type_path, _view)

                _seqs = os.listdir(_seq_dir)
                _seqs_num = len(_seqs)
                if (dataset == 'CASIA-B') or (dataset == 'OUMVLP') :
                    if len(pid_list) == 0:
                        if (len(list(set(label))) <= pid_num) and (_seqs_num >= frame_num):
                            seq_dir.append(_seq_dir)
                            label.append(_label)
                            seq_type.append(_seq_type)
                            view.append(_view)
                        elif (len(list(set(label))) > pid_num):
                            # For fair evaluation compared with other methods.
                            if ((data_type == 'silhouettes') and (_seqs_num >= 15)) or ((data_type == 'skeletons') and (_seqs_num > 0)):
                                # _seqs_num >= 15 for silhouettes-based methods.
                                # _seqs_num > 0 for skeleton-based methods. 
                                seq_dir.append(_seq_dir)
                                label.append(_label)
                                seq_type.append(_seq_type)
                                view.append(_view)
                    else:
                        if (_label in train_label) and (_seqs_num >= frame_num):
                            seq_dir.append(_seq_dir)
                            label.append(_label)
                            seq_type.append(_seq_type)
                            view.append(_view)
                        elif _label in test_label:
                            # For fair evaluation compared with other methods.
                            if ((data_type == 'silhouettes') and (_seqs_num >= 15)) or ((data_type == 'skeletons') and (_seqs_num > 0)):
                                # _seqs_num >= 15 for silhouettes-based methods.
                                # _seqs_num > 0 for skeleton-based methods. 
                                seq_dir.append(_seq_dir)
                                label.append(_label)
                                seq_type.append(_seq_type)
                                view.append(_view)

    return seq_dir, view, seq_type, label

def load_all_data(frame_num, pid_num, data_type, dataset, data_path, include_005, pid_list):
    if len(pid_list) == 2:
        train_label = pid_list[0]
        test_label = pid_list[1]

    sil_data_path = data_path[0]
    sil_label_list = sorted(list(os.listdir(sil_data_path)))

    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    for _label in sil_label_list:
        if include_005 is False:
            # In CASIA-B, data of subject #5 is incomplete.
            # Thus, we ignore it in training.
            if dataset == 'CASIA-B' and _label == '005':
                continue

        label_path = osp.join(sil_data_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                seq_info = [_label, _seq_type, _view]

                _seq_dir = [osp.join(_path, *seq_info) for _path in data_path]

                _seqs = os.listdir(_seq_dir[0])
                _seqs_num = len(_seqs) #silhouettes

                _ske_seqs = os.listdir(_seq_dir[1])
                _ske_seqs_num = len(_ske_seqs) #skeletons
                
                if (dataset == 'CASIA-B') or (dataset == 'OUMVLP') :
                    if len(pid_list) == 0:
                        if (len(list(set(label))) <= pid_num):
                            if (_seqs_num >= frame_num['silhouette']) and (_ske_seqs_num >= frame_num['skeleton']):
                                seq_dir.append(_seq_dir)
                                label.append(_label)
                                seq_type.append(_seq_type)
                                view.append(_view)
                        elif len(list(set(label))) > pid_num:
                            # For fair evaluation compared with other methods.
                            # frame_num >= 15 for silhouettes-based methods.
                            # frame_num > 0 for skeleton-based methods. 
                            if (_seqs_num >= 15) and (_ske_seqs_num > 0):
                                seq_dir.append(_seq_dir)
                                label.append(_label)
                                seq_type.append(_seq_type)
                                view.append(_view)
                    else:
                        if _label in train_label:
                            if (_seqs_num >= frame_num['silhouette']) and (_ske_seqs_num >= frame_num['skeleton']):
                                seq_dir.append(_seq_dir)
                                label.append(_label)
                                seq_type.append(_seq_type)
                                view.append(_view)
                        elif _label in test_label:
                            # For fair evaluation compared with other methods.
                            # frame_num >= 15 for silhouettes-based methods.
                            # frame_num > 0 for skeleton-based methods. 
                            if (_seqs_num >= 15) and (_ske_seqs_num > 0):
                                seq_dir.append(_seq_dir)
                                label.append(_label)
                                seq_type.append(_seq_type)
                                view.append(_view)

    return seq_dir, view, seq_type, label
