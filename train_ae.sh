#!/bin/bash

# LOAD_SNAPSHOT_DIR='./result/training/celeba/128x128/200/ae/20181110-133815-ae00(mse1.0-perc0.1)-(mwup0)/snapshot/epoch-150'
RESULT_DIR=./result/test

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_ae.sh $RESULT_DIR/copy

python3 ./src/train_ae.py \
--devices=1 \
\
--ae=ae00 \
\
--z_size=200 \
--static_z_size=200 \
--z_dout_rate=0.0 \
--z_mask_warm_up=100 \
\
--train_dataset=celeb \
--valid_dataset=celeb \
--test_dataset=celeb \
--train_set_path=./data/celeba/align-crop-128/train-valid \
--valid_set_path=./data/celeba/align-crop-128/test \
--test_set_path=./data/celeba/align-crop-128/test \
\
--batch_size=128 \
--img_size=128 \
--img_ch=3 \
--input_drop=0.0 \
\
--init_epoch=0 \
--max_epoch=200 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=20,50,100,200 \
--beta1=0.5 \
\
--perc_w=0.0 \
\
--valid_iter_intv=50 \
\
--save_snapshot_epochs=30,50,70,90,110,130,150,160,170,180,190,200 \
--result_dir=$RESULT_DIR \
# --load_snapshot_dir=$LOAD_SNAPSHOT_DIR \

