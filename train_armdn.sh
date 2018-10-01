#!/bin/bash

LOAD_SNAPSHOT_DIR=result/celeba/64x64/64/ae/20180928-134410-ae00/snapshot/epoch-100
RESULT_DIR=./result/celeba/64x64/64/armdn/`(date "+%Y%m%d-%H%M%S")`-armdn00

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_armdn.sh $RESULT_DIR/copy

python3 ./src/train_armdn.py \
--cur_device=0 \
\
--ae=ae00 \
--armdn=armdn00 \
--z_size=64 \
\
--n_gauss=50 \
--tau=1.0 \
\
--dataset=celeb \
--train_set_path=./data/celeba/align-crop-128/train-valid \
--valid_set_path=./data/celeba/align-crop-128/test \
\
--batch_size=128 \
--img_size=64 \
--img_ch=3 \
\
--init_epoch=0 \
--max_epoch=150 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=20,50,80 \
--beta1=0.5 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=50 \
\
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
--save_snapshot_epochs=100,130,150 \
--result_dir=$RESULT_DIR \
