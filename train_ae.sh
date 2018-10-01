#!/bin/bash

RESULT_DIR=./result/celeba/64x64/64/ae/`(date "+%Y%m%d-%H%M%S")`-ae00-mse1.0-perc0.1
# SNAPSHOT_DIR=result/celeba-32x32/32/lae/lae-satlins-0.0010-20180812-162033/snapshot/epoch-120

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_ae.sh $RESULT_DIR/copy

python3 ./src/train_ae.py \
--devices=1 \
\
--ae=ae00 \
--z_size=64 \
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
--max_epoch=200 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=100,150 \
--beta1=0.5 \
\
--mse_w=1.0 \
--perc_w=0.1 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=50 \
\
--result_dir=$RESULT_DIR \
--save_snapshot_epochs=0,50,100,150,200 \
# --load_snapshot_dir=$SNAPSHOT_DIR \
