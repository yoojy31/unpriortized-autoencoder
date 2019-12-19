#!/bin/bash

python3 ./src/split_dataset.py \
\
--dataset=mnist1 \
--train_set_path=./data/mnist/train-images.idx3-ubyte \
--test_set_path=./data/mnist/t10k-images.idx3-ubyte \
--split_train_idx=55000 \
--result_dir=./data/mnist2 \
