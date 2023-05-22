from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--ckp_prefix', default=None, type=str,
                    help='ckp_prefix: prefix of the checkpoint to load. Default: None')
parser.add_argument('--model', default='fuse', type=str,
                    help='model: model for test. Default: fuse')
parser.add_argument('--iter', default='100000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()

if opt.model == 'fuse':
    from configs.config import conf
    part_dim = conf['model']['fuse_cfg']['out_channels']
elif (opt.model == 'cnn_gaitpart') or (opt.model == 'cnn_gaitgl'):
    from configs.config_cnn import conf
    part_dim = conf['model']['model_cfg']['out_channels']
elif (opt.model == 'msgg') or (opt.model == 'msgg1layer') or (opt.model == 'msgg2layer') or (opt.model == 'osgg') or (opt.model == 'st_gcn'):
    from configs.config_msgg import conf
    part_dim = False

print("conf:", conf)
print("opt:", opt)
dataset = conf['dataset'].split('_')[0]

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    if dataset == 'CASIA-B':
        view_num = 11
    elif dataset == 'OUMVLP':
        view_num = 14
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / (view_num -1.0)
    if not each_angle:
        result = np.mean(result)
    return result

def include_diag(acc):
    if dataset == 'CASIA-B':
        view_num = 11
    elif dataset == 'OUMVLP':
        view_num = 14
    result = np.sum(acc, 1) / view_num
    return result


m = initialization(conf, model=opt.model, state='test', test=opt.cache)[0]

# load model checkpoint of iteration opt.iter
# print('Loading the model of iteration %d...' % opt.iter)
# m.load(opt.iter)

if opt.ckp_prefix is None:
    print('Loading the model of iteration %d...' % opt.iter)
    m.load(opt.iter)
else:
    m.load_ckp(opt.ckp_prefix)

print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)
print('Transformation complete. Cost:', datetime.now() - time)
print('Evaluating...')
acc = evaluation(test, dataset, part_dim)
print('acc.shape:', acc.shape)
print('Evaluation complete. Cost:', datetime.now() - time)
# Print rank-1 accuracy, including identical-view cases
for i in range(1):
    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    if dataset == 'CASIA-B':
        print('NM: %.3f,\tBG: %.3f,\tCL:%.3f'%(
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i])))
    elif dataset == 'OUMVLP':
        print(np.mean(acc[0, :, :, i]))
# Print rank-1 accuracy, excluding identical-view cases
for i in range(1):
    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
    if dataset == 'CASIA-B':
        print('NM: %.3f,\tBG: %.3f,\tCL:%.3f'%(
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i])))
    elif dataset == 'OUMVLP':
        print(de_diag(acc[0, :, :, i]))
# Print rank-1 accuracy (Each Angle, excluding identical-view cases)
np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):
    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
    if dataset == 'CASIA-B':
        print('NM:', de_diag(acc[0, :, :, i], True))
        print('BG:', de_diag(acc[1, :, :, i], True))
        print('CL:', de_diag(acc[2, :, :, i], True))
    if dataset == 'OUMVLP':
        print(de_diag(acc[0, :, :, i], True))
# Print rank-1 accuracy (Each Angle, including identical-view cases)
for i in range(1):
    print('===Rank-%d of each angle (Include identical-view cases)===' % (i + 1))
    if dataset == 'CASIA-B':
        print('NM:', include_diag(acc[0, :, :, i]))
        print('BG:', include_diag(acc[1, :, :, i]))
        print('CL:', include_diag(acc[2, :, :, i]))
    elif dataset == 'OUMVLP':
        print(include_diag(acc[0, :, :, i]))