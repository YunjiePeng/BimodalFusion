conf = {
    "WORK_PATH": "./work/cnn_silhouette",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "dataset": "CASIA-B", # "CASIA-B" or "OUMVLP"
    "model": {
        'model_cfg':{
            #---Settings for CASIA-B---
            'hidden_dim': 128,
            'out_channels': 128,

            #---Settings for OU-MVLP---
            # 'hidden_dim': 256,
            # 'out_channels': 256,
        },
        'lr': 1e-4,
        'margin': 0.2,
        'frame_num': 30,
        'frame_num_min': 15,
        'num_workers': 24,
        'restore_iter': 0,
        'hard_or_full_trip': 'full',
        #---Settings for GaitGL on CASIA-B---
        'batch_size': (8, 16),
        'milestones': [70000],
        'total_iter': 80000,

        #---Settings for GaitGL on OU-MVLP---
        # 'batch_size': (32, 8),
        # 'milestones': [150000, 200000],
        # 'total_iter': 210000,

        #---Settings for GaitPart on CASIA-B---
        # 'batch_size': (8, 16),
        # 'milestones': [100000],
        # 'total_iter': 120000,

        #---Settings for GaitPart on OU-MVLP---
        # 'batch_size': (32, 16),
        # 'milestones': [150000],
        # 'total_iter': 250000,

        #---Trained Models---
        'model_name': 'gaitgl_casia-b',
        # 'model_name': 'gaitgl_oumvlp',
        # 'model_name': 'gaitpart_casia-b',
        # 'model_name': 'gaitpart_oumvlp',
    },
    'data': {
        'data_type': 'silhouettes',
        'CASIA-B': {
            'resolution': '64',
            'pid_num': 73,
            'pid_shuffle': False,
            'data_path': ["/home/pengyunjie/data/casia-b/silhouettes64/"],
        },
        'OUMVLP': {
            'resolution': '64',
            'pid_num': 5153,
            'pid_shuffle': False,
            'data_path': ["/home/pengyunjie/dataset/ou-mvlp/silhouettes64/"],
        },
    },
}





