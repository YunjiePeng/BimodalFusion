import os
import torch
import random
import logging
import numpy as np

from copy import deepcopy
from .utils import load_data

from .model_cnn_gaitpart import Model_CNN_GaitPart
from .model_cnn_gaitgl import Model_CNN_GaitGL

from .model_msgg import Model_MSGG
from .model_osgg import Model_OSGG
from .model_msgg1layer import Model_MSGG1Layer
from .model_msgg2layer import Model_MSGG2Layer
from .model_st_gcn import Model_ST_GCN

from .model import Model


def set_seed(local_rank):
    SEED=2020 + local_rank
    # SEED=1234 + local_rank
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def initialization(config, train=False, test=False, model='model', state='train', local_rank=0):
    logging.info("Initialzing...")
    set_seed(local_rank)
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, test_source = initialize_data(config, train, test)

    config['model'].pop('frame_num_min')
    if model == 'fuse':
        return initialize_model(config, train_source, test_source, state, local_rank)
    elif (model == 'cnn_gaitset_bin16') or (model == 'cnn_gaitpart') or (model == 'cnn_gaitgl'):
        return initialize_silhouette_model(config, model, train_source, test_source, state, local_rank)
    elif (model == 'msgg') or (model == 'osgg') or (model == 'msgg1layer') or (model == 'msgg2layer') or (model == 'st_gcn'):
        return initialize_skeleton_model(config, model, train_source, test_source, state, local_rank)

def initialize_data(config, train=False, test=False):
    logging.info("Initializing data source...")
    dataset = config['dataset']
    data_conf = config['data'][dataset]
    data_type = config['data']['data_type']

    frame_num_min = config['model']['frame_num_min']
    train_source, test_source = load_data(dataset, data_type, frame_num_min, **data_conf, cache=(train or test))

    # frame_num = config['model']['frame_num']
    # train_source, test_source = load_data(dataset, data_type, frame_num, **data_conf, cache=(train or test))
    if train:
        logging.info("Loading training data...")
        train_source.load_all_data()
    if test:
        logging.info("Loading test data...")
        test_source.load_all_data()
    logging.info("Data initialization complete.")
    return train_source, test_source

def initialize_model(config, train_source, test_source, state, local_rank):
    logging.info("Initializing model...")
    dataset = config['dataset']
    data_conf = config['data'][dataset]

    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_conf['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        dataset,
        data_conf['pid_num'],
        data_conf['pid_shuffle'],
        model_config['cnn_cfg']['out_channels'],
        model_config['pgcn_cfg']['graph_cfg']['layout'],
        model_config['pgcn_cfg']['graph_cfg']['strategy'],
        model_config['pgcn_cfg']['in_channels'],
        model_config['pgcn_cfg']['out_channels'],
        model_config['hard_or_full_trip'],
        batch_size,
        model_config['frame_num'],
        model_config['margin'],
    ]))
    model_param['state'] = state
    model_param['local_rank'] = local_rank

    m = Model(**model_param)
    logging.info("Model initialization complete.")
    return m, model_param['save_name']

def initialize_silhouette_model(config, model, train_source, test_source, state, local_rank):
    logging.info("Initializing model...")
    dataset = config['dataset']
    data_conf = config['data'][dataset]

    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_conf['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        dataset,
        data_conf['pid_num'],
        data_conf['pid_shuffle'],
        model_config['model_cfg']['hidden_dim'],
        model_config['model_cfg']['out_channels'],
        model_config['hard_or_full_trip'],
        batch_size,
        model_config['frame_num'],
        model_config['margin'],
    ]))
    model_param['state'] = state
    model_param['local_rank'] = local_rank

    if model == 'cnn_gaitset_bin16':
        m = Model_CNN_GaitSet_Bin16(**model_param)
    elif model == 'cnn_gaitpart':
        m = Model_CNN_GaitPart(**model_param)
    elif model == 'cnn_gaitgl':
        m = Model_CNN_GaitGL(**model_param)
    
    logging.info("Model initialization complete.")
    return m, model_param['save_name']

def initialize_skeleton_model(config, model, train_source, test_source, state, local_rank):
    logging.info("Initializing model...")
    dataset = config['dataset']
    data_conf = config['data'][dataset]

    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_conf['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        dataset,
        data_conf['pid_num'],
        data_conf['pid_shuffle'],
        model_config['model_cfg']['graph_cfg']['layout'],
        model_config['model_cfg']['graph_cfg']['strategy'],
        model_config['model_cfg']['in_channels'],
        model_config['model_cfg']['out_channels'],
        model_config['hard_or_full_trip'],
        batch_size,
        model_config['frame_num'],
        model_config['margin'],
    ]))
    model_param['state'] = state
    model_param['local_rank'] = local_rank

    if model == 'msgg':
        m = Model_MSGG(**model_param)
    elif model == 'osgg':
        m = Model_OSGG(**model_param)
    elif model == 'msgg1layer':
        m = Model_MSGG1Layer(**model_param)
    elif model == 'msgg2layer':
        m = Model_MSGG2Layer(**model_param)
    elif model == 'st_gcn':
        m = Model_ST_GCN(**model_param)
        
    logging.info("Model initialization complete.")
    return m, model_param['save_name']

# def initialize_pgg_model(config, model, train_source, test_source, state, local_rank):
#     logging.info("Initializing model...")
#     dataset = config['dataset']
#     data_conf = config['data'][dataset]

#     model_config = config['model']
#     model_param = deepcopy(model_config)
#     model_param['train_source'] = train_source
#     model_param['test_source'] = test_source
#     model_param['train_pid_num'] = data_conf['pid_num']
#     batch_size = int(np.prod(model_config['batch_size']))
#     model_param['save_name'] = '_'.join(map(str,[
#         model_config['model_name'],
#         dataset,
#         data_conf['pid_num'],
#         data_conf['pid_shuffle'],
#         model_config['model_cfg']['graph_cfg']['layout'],
#         model_config['model_cfg']['graph_cfg']['strategy'],
#         model_config['model_cfg']['in_channels'],
#         model_config['model_cfg']['out_channels'],
#         model_config['hard_or_full_trip'],
#         batch_size,
#         model_config['frame_num'],
#         model_config['margin'],
#     ]))
#     model_param['state'] = state
#     model_param['local_rank'] = local_rank

#     if model == 'pgg':
#         m = Model_PGG(**model_param)
#     elif model == 'pgg1layer':
#         m = Model_PGG1Layer(**model_param)
#     elif model == 'pgg2layer':
#         m = Model_PGG2Layer(**model_param)
        
#     logging.info("Model initialization complete.")
#     return m, model_param['save_name']

# def initialize_st_gcn_model(config, train_source, test_source, state, local_rank):
#     logging.info("Initializing model...")
#     dataset = config['dataset']
#     data_conf = config['data'][dataset]

#     model_config = config['model']
#     model_param = deepcopy(model_config)
#     model_param['model_cfg'].pop('num_id')

#     model_param['train_source'] = train_source
#     model_param['test_source'] = test_source
#     model_param['train_pid_num'] = data_conf['pid_num']
#     batch_size = int(np.prod(model_config['batch_size']))
#     model_param['save_name'] = '_'.join(map(str,[
#         model_config['model_name'],
#         dataset,
#         data_conf['pid_num'],
#         data_conf['pid_shuffle'],
#         model_config['model_cfg']['graph_cfg']['layout'],
#         model_config['model_cfg']['graph_cfg']['strategy'],
#         model_config['model_cfg']['in_channels'],
#         model_config['model_cfg']['out_channels'],
#         model_config['hard_or_full_trip'],
#         batch_size,
#         model_config['frame_num'],
#         model_config['margin'],
#     ]))
#     model_param['state'] = state
#     model_param['local_rank'] = local_rank

#     m = Model_ST_GCN(**model_param)
#     logging.info("Model initialization complete.")
#     return m, model_param['save_name']
