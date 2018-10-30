#!/bin/bash

RESULT_DIR=./result/training/celeba/64x64/512/ae/`(date "+%Y%m%d-%H%M%S")`-ae11'(512-64*0.3)'-mse1.0
# SNAPSHOT_DIR=result/celeba/128x128/64/ae/20181002-140002-ae00-mse1.0-perc1.0/snapshot/epoch-50

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_ae.sh $RESULT_DIR/copy

python3 ./src/train_ae.py \
--devices=1 \
\
--ae=ae11 \
--z_size=512 \
--static_z_size=64 \
--z_dout_rate=0.3 \
\
--train_dataset=celeb \
--valid_dataset=celeb \
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
--lr_decay_epochs=60 \
--beta1=0.5 \
\
--mse_w=1.0 \
--perc_w=0.0 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=10 \
\
--result_dir=$RESULT_DIR \
--save_snapshot_epochs=0,30,50,55,60,70,100 \
# --load_snapshot_dir=$SNAPSHOT_DIR \
