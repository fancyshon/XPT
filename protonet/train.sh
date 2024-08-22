#!/bin/bash

python protonet/protonet.py --n_way 5 --n_shot 5 --dataset CUB --batch_size 256 >> protonet/cub_log
python protonet/protonet.py --n_way 5 --n_shot 5 --dataset miniImageNet --batch_size 256 >> protonet/miniimagenet_log
python protonet/protonet.py --n_way 5 --n_shot 5 --dataset tieredImageNet --batch_size 256 >> protonet/tieredimagenet_log

# python protonet/protonet.py --n_way 5 --n_shot 1 --dataset CUB --batch_size 256 >> protonet/cub_log
# python protonet/protonet.py --n_way 5 --n_shot 1 --dataset miniImageNet --batch_size 256 >> protonet/miniimagenet_log
# python protonet/protonet.py --n_way 5 --n_shot 1 --dataset tieredImageNet --batch_size 256 >> protonet/tieredimagenet_log