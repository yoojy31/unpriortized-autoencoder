#!/bin/bash

# --patch_drop \
# --ordering
# LOAD_SNAPSHOT_DIR='result/training/celeba/64x64/256/ae/20181031-222508-ae11-(64-0.5)-mse1.0-perc0.1/snapshot/epoch-100'
RESULT_DIR=./result/training/cifar10/32x32/100/ae/`(date "+%Y%m%d-%H%M%S")`-'ae00(mse1.0)-(mwup0)'

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_ae.sh $RESULT_DIR/copy

python3 ./src/train_ae.py \
--devices=0 \
\
--ae=ae00 \
\
--z_size=100 \
--static_z_size=100 \
--z_dout_rate=0.0 \
--z_mask_warm_up=0 \
\
--train_dataset=cifar \
--valid_dataset=cifar \
--train_set_path=./data/cifar10/train \
--valid_set_path=./data/cifar10/test \
\
--batch_size=64 \
--img_size=32 \
--img_ch=3 \
--input_drop=0.0 \
\
--init_epoch=0 \
--max_epoch=80 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=20,50 \
--beta1=0.5 \
\
--perc_w=0.1 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=50 \
\
--save_snapshot_epochs=50,60,70,80 \
--result_dir=$RESULT_DIR \
# --load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
