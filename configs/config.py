conf = {
    "WORK_PATH": "./work/bimodal_fusion",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    'dataset': "CASIA-B", # "CASIA-B" or "OUMVLP"
    "model": {
        #---Settings for CASIA-B---
        'pretrain_gcn': 'pretrain_gcn/msgg_casia-b_hrnet_gait-temporal_CASIA-B_HRNET_73_False_body_12_gait_temporal_3_64_full_128_30_0.2-100000-encoder.ptm',
        'pretrain_cnn': 'pretrain_cnn/gaitgl_casia-b_CASIA-B_73_False_128_128_full_128_30_0.2-80000-encoder.ptm',
        # 'pretrain_cnn': 'pretrain_cnn/gaitpart_casia-b_CASIA-B_73_False_128_128_full_128_30_0.2-120000-encoder.ptm',
        
        #---Settings for OU-MVLP---
        # 'pretrain_gcn': 'pretrain_gcn/msgg_oumvlp_alphapose_gait-temporal_OUMVLP_ALPHAPOSE_5153_False_body_12_gait_temporal_3_128_full_512_18_0.2-300000-encoder.ptm',
        # 'pretrain_cnn': 'pretrain_cnn/gaitgl_oumvlp_OUMVLP_5153_False_256_256_full_256_30_0.2-210000-encoder.ptm',
        # 'pretrain_cnn': 'pretrain_cnn/gaitpart_oumvlp_OUMVLP_5153_False_256_256_full_512_30_0.2-250000-encoder.ptm',

        'cnn_branch': "gaitgl", # ["gaitgl" or "gaitpart"]
        'cnn_cfg':{
            #---Settings for CASIA-B---
            'hidden_dim': 128,
            'out_channels': 128,

            #---Settings for OUMVLP---
            # 'hidden_dim': 256,
            # 'out_channels': 256,
        },
        'pgcn_cfg':{
            'in_channels': 3,
            'graph_cfg': {
                'layout': 'body_12',
                'strategy': 'gait_temporal',
            },
            'edge_importance_weighting': True,
            #---Settings for CASIA-B---
            'out_channels': 64,
            'num_id': 73,

            #---Settings for OUMVLP---
            # 'out_channels': 128,
            # 'num_id': 5153,
        },
        'fuse_cfg':{
            'part_num': 64, # [ 64 for gaitgl, 16 for gaitpart]
            #---Settings for CASIA-B---
            'out_channels': 128,

            #---Settings for OUMVLP---
            # 'out_channels': 256,
        },
        'lr': 0.1,
        'margin': 0.2,
        'frame_num': {
            'silhouette': 30,
            'skeleton': 30, # for CASIA-B
            # 'skeleton': 18, # for OUMVLP
        },
        'frame_num_min': {
            'silhouette': 15,
            'skeleton': 30, # for CASIA-B
            # 'skeleton': 18, # for OUMVLP
        },
        'num_workers': 24,
        'restore_iter': 0,
        'hard_or_full_trip': 'full',
        #---Settings for CASIA-B---
        'batch_size': (8, 16),
        'step_iter': 2000,
        'total_iter': 10000,

        #---Settings for OUMVLP---
        # 'batch_size': (32, 8),
        # 'step_iter': 4000,
        # 'total_iter': 20000,
        
        #---Trained Models---
        'model_name': 'bimodal_fusion_gaitgl_casia-b_hrnet',
        # 'model_name': 'bimodal_fusion_gaitgl_oumvlp_alphapose',
        # 'model_name': 'bimodal_fusion_gaitpart_casia-b_hrnet',
        # 'model_name': 'bimodal_fusion_gaitpart_oumvlp_alphapose',
    },
    'data': {
        'data_type': 'all',
        'CASIA-B': {
            'resolution': '64',
            'pid_num': 73,
            'pid_shuffle': False,
            'data_path':[
                "/home/pengyunjie/data/casia-b/silhouettes64/", # silhouette data path.
                "/home/pengyunjie/data/casia-b/pose/hrnet/pyramid_keypoints", # skeleton data path (HRNET).
                # "/home/pengyunjie/dataset/casia-b/pose/openpose/pyramid_keypoints", # skeleton data path (Openpose).
            ],
        },
        'OUMVLP': {
            'resolution': '64',
            'pid_num': 5153,
            'pid_shuffle': False,
            'data_path': [
                "/data_1/pengyunjie/dataset/ou-mvlp/silhouettes/", # silhouettes
                "/data_1/pengyunjie/dataset/ou-mvlp/pose/pyramid_alpha_pose_npy", # skeletons
            ],
        },
    },
}





