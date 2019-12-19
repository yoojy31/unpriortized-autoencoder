#!/bin/bash

python3 ./src/augment_bsds.py \
\
--dataset=bsds \
--train_set_path=/mnt/storage/datasets/bsds/bsds300/image/train \
--valid_set_path=/mnt/storage/datasets/bsds/bsds300/image/train \
--test_set_path=/mnt/storage/datasets/bsds/bsds300/image/test \
--result_dir=/home/yoojy/Datasets/bsds300/augmented \
