import os
import os.path as osp
from scipy import misc as scisc
import cv2
import numpy as np
import argparse
import logging

# parser = argparse.ArgumentParser(description='Silhouette_Preprocess')
# parser.add_argument('--input_path', default="/home/pengyunjie/data/casia-b/silhouettes/", type=str,
#                     help='Root path of raw silhoutte dataset.')
# parser.add_argument('--output_path', default="/home/pengyunjie/data/casia-b/silhouettes64/", type=str,
#                     help='Root path for output.')
# parser.add_argument('--log_file', default="/home/pengyunjie/data/casia-b/log_silhouette_preprocess.txt", type=str,
#                     help='Log file path.')
# args = parser.parse_args()

parser = argparse.ArgumentParser(description='Silhouette_Preprocess')
parser.add_argument('--input_path', default="/data_1/pengyunjie/dataset/fvg/fvg_new/silhouettes/", type=str,
                    help='Root path of raw silhoutte dataset.')
parser.add_argument('--output_path', default="/data_1/pengyunjie/dataset/fvg/fvg_new/silhouettes64/", type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default="/data_1/pengyunjie/dataset/fvg/fvg_new/log_silhouette_preprocess.txt", type=str,
                    help='Log file path.')
args = parser.parse_args()

def align_and_resize(sil_path, sil_height=64, sil_width=64):
    sil_img = cv2.imread(sil_path)[:, :, 0]

    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if sil_img.sum() <= 10000:
        log_str = 'Too little white pixels:{}'.format(sil_path)
        with open(args.log_file, 'a') as log_f:
            log_f.write(log_str)
        logging.info(log_str)
        return False, None

    # Get the top and bottom point
    y = sil_img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    sil_img = sil_img[y_top:y_btm + 1, :]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = sil_img.shape[1] / sil_img.shape[0]
    _sil_width = int(sil_height * _r)
    sil_img = cv2.resize(sil_img, (_sil_width, sil_height), interpolation=cv2.INTER_CUBIC)

    # Get the median of x axis and regard it as the x center of the person.
    sum_point = sil_img.sum()
    sum_column = sil_img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        log_str = 'No center:{}'.format(frame_name)
        with open(args.log_file, 'a') as log_f:
            log_f.write(log_str)
        logging.info(log_str)
        return False, None

    h_sil_width = int(sil_width / 2)
    left = x_center - h_sil_width
    right = x_center + h_sil_width
    if left <= 0 or right >= sil_img.shape[1]:
        left += h_sil_width
        right += h_sil_width
        _ = np.zeros((sil_img.shape[0], h_sil_width), dtype=np.uint8)
        sil_img = np.concatenate([_, sil_img, _], axis=1)

    sil_img = sil_img[:, left:right]
    return True, sil_img

label_list = sorted(os.listdir(args.input_path))
# Walk the input path
for _label in label_list:
    type_list = sorted(os.listdir(osp.join(args.input_path, _label)))
    for _type in type_list:
        view_list = sorted(os.listdir(osp.join(args.input_path, _label, _type)))
        for _view in view_list:
            seq_info = [_label, _type, _view]

            save_dir = os.path.join(args.output_path, *seq_info)
            if osp.exists(save_dir) is False:
                os.makedirs(save_dir)

            sil_dir = osp.join(args.input_path, *seq_info)
            sil_list = sorted(os.listdir(sil_dir))
            for _sil in sil_list:
                sil_path = osp.join(sil_dir, _sil)
                success, sil_pretreated = align_and_resize(sil_path)
                if success:
                    cv2.imwrite(osp.join(save_dir, _sil), sil_pretreated)
            print("FINISHED:{}-{}-{}".format(_label, _type, _view))

            