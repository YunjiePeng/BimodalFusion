import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = sorted(list(set(self.label)))
        self.seq_type_set = sorted(list(set(self.seq_type)))
        self.view_set = sorted(list(set(self.view)))
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __imgloader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    def __keypointsloader__(self, path):
            return self.keypoints2xarray(
                path)[:, :, :].astype('float32')

    def __getitem__(self, index):
        # pose sequence
        data = []
        frame_set = []
        if not self.cache:
            sil_data = [self.__imgloader__(self.seq_dir[index][0])]
            ske_data = [self.__imgloader__(self.seq_dir[index][1])]
            keypoints_data = [self.__keypointsloader__(self.seq_dir[index][2])]
            data = [sil_data, ske_data, keypoints_data]
            sil_frame_set = [set(feature.coords['frame'].values.tolist()) for feature in sil_data]
            sil_frame_set = list(set.intersection(*sil_frame_set))
            ske_frame_set = [set(feature.coords['frame'].values.tolist()) for feature in ske_data]
            ske_frame_set = list(set.intersection(*ske_frame_set))
            keypoints_frame_set = [set(feature.coords['frame'].values.tolist()) for feature in keypoints_data]
            keypoints_frame_set = list(set.intersection(*keypoints_frame_set))
            frame_set = [sil_frame_set, ske_frame_set, keypoints_frame_set]

        elif self.data[index] is None:
            sil_data = [self.__imgloader__(self.seq_dir[index][0])]
            ske_data = [self.__imgloader__(self.seq_dir[index][1])]
            keypoints_data = [self.__keypointsloader__(self.seq_dir[index][2])]
            data = [sil_data, ske_data, keypoints_data]
            sil_frame_set = [set(feature.coords['frame'].values.tolist()) for feature in sil_data]
            sil_frame_set = list(set.intersection(*sil_frame_set))
            ske_frame_set = [set(feature.coords['frame'].values.tolist()) for feature in ske_data]
            ske_frame_set = list(set.intersection(*ske_frame_set))
            keypoints_frame_set = [set(feature.coords['frame'].values.tolist()) for feature in keypoints_data]
            keypoints_frame_set = list(set.intersection(*keypoints_frame_set))
            frame_set = [sil_frame_set, ske_frame_set, keypoints_frame_set]
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],

    def keypoints2xarray(self, file_path):
        skeletons = sorted(list(os.listdir(file_path)))
        frame_list = [np.load(open(osp.join(file_path, _skeleton_path), 'rb'))
                      for _skeleton_path in skeletons
                      if osp.isfile(osp.join(file_path, _skeleton_path))]
        num_list = sorted(list(range(len(frame_list))))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'keypoints', 'features'],
        )
        return data_dict

    def img2xarray(self, file_path):
        imgs = sorted(list(os.listdir(file_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(file_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(file_path, _img_path))]
        ######################
        num_list = sorted(list(range(len(frame_list))))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)
