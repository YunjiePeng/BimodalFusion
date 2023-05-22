conf = {
    "WORK_PATH": "./work/gcn_skeleton", # ["./work/gcn_skeleton" or "./work/ablation_study"]
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    'dataset': "CASIA-B_HRNET", # ["CASIA-B_HRNET" or "CASIA-B_OPENPOSE" or "OUMVLP_ALPHAPOSE" or "OUMVLP_OPENPOSE"]
    "model": {
        'model_cfg':{
            'in_channels': 3, # 3 for (x, y, confidence score)
            'graph_cfg': {
                'layout': 'body_12',
                'strategy': 'gait_temporal', # ['gait_temporal', 'spatial', 'uniform', 'distance']
            },
            'edge_importance_weighting': True,
            #---Settings for CASIA-B---
            'out_channels': 64,
            'num_id': 73,

            #---Settings for OUMVLP_ALPHAPOSE---
            # 'out_channels': 128,
            # 'num_id': 5153,
        },
        'lr': 0.1,
        'margin': 0.2,
        'num_workers': 24,
        'restore_iter': 0,    
        'hard_or_full_trip': 'full',
        #---Settings for CASIA-B---
        'frame_num': 30,
        'frame_num_min': 30,
        'step_iter': 25000,
        'total_iter': 100000,
        'batch_size': (8, 16),

        #---Settings for OUMVLP_ALPHAPOSE---
        # 'frame_num': 18,
        # 'frame_num_min': 18,
        # 'step_iter': 75000,
        # 'total_iter': 300000,
        # 'batch_size': (32, 16),

        #---Trained Models---
        #===For "WORK_PATH": "./work/gcn_skeleton"===
        'model_name': 'msgg_casia-b_hrnet_gait-temporal',
        # 'model_name': 'msgg_casia-b_openpose_gait-temporal',
        # 'model_name': 'msgg_oumvlp_alphapose_gait-temporal',
        # 'model_name': 'st-gcn_casia-b_hrnet_spatial',

        #===For "WORK_PATH": "./work/ablation_study"===
        # 'model_name': 'msgg_casia-b_hrnet_spatial',
        # 'model_name': 'msgg_casia-b_hrnet_uniform',
        # 'model_name': 'msgg_casia-b_hrnet_distance',
        # 'model_name': 'msgg_casia-b_hrnet_gait-temporal_1layer', # Joints
        # 'model_name': 'msgg_casia-b_hrnet_gait-temporal_2layer', # Joints+Limbs
        # 'model_name': 'msgg_casia-b_hrnet_gait-temporal_separate', # Joints+Limbs+Bodyparts (Separate)
        # 'model_name': 'osgg_casia-b_hrnet_gait-temporal', # One Scale Only: Joints+Joints+Joints
    },
    'data': {
        'data_type': 'skeletons',
        'CASIA-B_HRNET': {
            'resolution': '64',
            'pid_num': 73,
            'pid_shuffle': False,
            'data_path': ["/home/pengyunjie/data/casia-b/pose/hrnet/pyramid_keypoints"],
        },
        'CASIA-B_OPENPOSE': {
            'resolution': '64',
            'pid_num': 73,
            'pid_shuffle': False,
            'data_path': ["/home/pengyunjie/data/casia-b/pose/openpose/pyramid_keypoints"],
        },
        'OUMVLP_ALPHAPOSE': {
            'resolution': '64',
            'pid_num': 5153,
            'pid_shuffle': False,
            'data_path': ["/home/pengyunjie/data/ou-mvlp/pose/pyramid_alpha_pose_npy"],
        },
        'OUMVLP_OPENPOSE': {
            'resolution': '64',
            'pid_num': 5153,
            'pid_shuffle': False,
            'data_path': ["/home/pengyunjie/dataset/ou-mvlp/pose/pyramid_open_pose_npy"],
        },
    },
}





