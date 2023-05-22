import os
import os.path as osp

import numpy as np
import pickle

from .data_set import DataSet
from warnings import warn


def load_data(frame_num, data_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    silimg_path = data_path['silimg_path']
    skeimg_path = data_path['skeimg_path']
    keypoints_path = data_path['keypoints_path']
    print("silimg_path={}, skeimg_path={}, keypoints_path={}, cache={}".format(silimg_path, skeimg_path, keypoints_path, cache))
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    for _label in sorted(list(os.listdir(silimg_path))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
            continue
        sil_label_path = osp.join(silimg_path, _label)
        ske_label_path = osp.join(skeimg_path, _label)
        keypoints_label_path = osp.join(keypoints_path, _label)
        for _seq_type in sorted(list(os.listdir(sil_label_path))):
            sil_seq_type_path = osp.join(sil_label_path, _seq_type)
            ske_seq_type_path = osp.join(ske_label_path, _seq_type)
            keypoints_seq_type_path = osp.join(keypoints_label_path, _seq_type)
            for _view in sorted(list(os.listdir(sil_seq_type_path))):
                _sil_seq_dir = osp.join(sil_seq_type_path, _view)
                _ske_seq_dir = osp.join(ske_seq_type_path, _view)
                _keypoints_seq_dir = osp.join(keypoints_seq_type_path, _view)

                sil_seqs = os.listdir(_sil_seq_dir)
                sil_imgs_num = len(sil_seqs)

                if sil_imgs_num >= frame_num:
                    seq_dir.append([_sil_seq_dir, _ske_seq_dir, _keypoints_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)



    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))
    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)

    pid_list = np.load(pid_fname)
    train_list = pid_list[0]
    test_list = pid_list[1]
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        cache, resolution)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution)

    return train_source, test_source
