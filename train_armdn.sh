#!/bin/bash

LOAD_SNAPSHOT_DIR=result/celeba/128x128/256/ae/20181003-155039-ae00-mse1.0-perc1.0-\(resave\)/snapshot/epoch-50
RESULT_DIR=./result/celeba/128x128/256/armdn/`(date "+%Y%m%d-%H%M%S")`-armdn00-ae00-perc

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_armdn.sh $RESULT_DIR/copy

python3 ./src/train_armdn.py \
--devices=1 \
\
--ae=ae00 \
--armdn=armdn00 \
--z_size=64 \
\
--n_gauss=20 \
--tau=1.0 \
\
--dataset=celeb \
--train_set_path=./data/celeba/align-crop-128/train-valid \
--valid_set_path=./data/celeba/align-crop-128/test \
\
--batch_size=128 \
--img_size=128 \
--img_ch=3 \
\
--init_epoch=0 \
--max_epoch=70 \
--lr=2e-4 \
--lr_decay_rate=2e-1 \
--lr_decay_epochs=20,40 \
--beta1=0.5 \
\
--perc_w=0.0 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=50 \
\
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
--save_snapshot_epochs=40,50,60,70 \
--result_dir=$RESULT_DIR \
