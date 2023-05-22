from model.initialization import initialization
import torch.distributed as dist
import argparse
import logging
import torch
import random
import numpy as np

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--model', default='fuse', type=str,
                    help='model: model for train. Default: fuse')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the training data will be loaded at once'
                         ' before the training start. Default: TRUE')
opt = parser.parse_args()

if opt.model == 'fuse':
    from configs.config import conf
elif (opt.model == 'cnn_gaitpart') or (opt.model == 'cnn_gaitgl'):
    from configs.config_cnn import conf
elif (opt.model == 'msgg') or (opt.model == 'osgg') or (opt.model == 'msgg1layer') or (opt.model == 'msgg2layer') or (opt.model == 'st_gcn'):
    from configs.config_msgg import conf

logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
logging.info(conf)

m = initialization(conf, model=opt.model, state='train', train=opt.cache, local_rank=opt.local_rank)[0]

logging.info("Training START")
m.fit()
dist.destroy_process_group()
logging.info("Training COMPLETE")
