1.Train
'''
train.py
nproc_per_node: the number of gpus
'''
# Silhouette Branch
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py --model=cnn_gaitpart
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py --model=cnn_gaitgl
# Skeleton Branch
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py --model=msgg
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py --model=osgg
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py --model=st_gcn
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py --model=msgg1layer
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py --model=msgg2layer
# BiFusion
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 train.py --model=fuse

2.Test
'''
test_parallel.py
Parallel test, support for ((--model != msgg1layer) and (--model != msgg2layer))
Require (batch_size == nproc_per_node) is True.
'''
# Silhouette Branch
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 test_parallel.py --model=cnn_gaitgl --batch_size=4 --iter=80000
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 test_parallel.py --model=cnn_gaitpart --batch_size=4 --iter=80000
# Skeleton Branch
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 test_parallel.py --model=msgg --batch_size=4 --iter=100000
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 test_parallel.py --model=osgg --batch_size=4 --iter=100000
# BiFusion
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 test_parallel.py --model=fuse --batch_size=8 --iter=10000

[test.py]
python test.py --model=msgg1layer --iter=100000
python test.py --model=msgg2layer --iter=100000
# Support for ((--model == msgg1layer) or (--model == msgg1layer))

3.Stop Parallel Training/Testing
kill $(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "test.py" | grep -v grep | awk '{print $2}')