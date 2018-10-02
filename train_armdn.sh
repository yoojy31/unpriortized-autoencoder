#!/bin/bash

LOAD_SNAPSHOT_DIR=result/celeba/128x128/128/ae/20181002-024837-ae01-mse1.0-perc1.0/snapshot/epoch-50
RESULT_DIR=./result/celeba/128x128/128/armdn/`(date "+%Y%m%d-%H%M%S")`-armdn00-ae00-perc

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_armdn.sh $RESULT_DIR/copy

python3 ./src/train_armdn.py \
--devices=1 \
\
--ae=ae00 \
--armdn=armdn00 \
--z_size=128 \
\
--n_gauss=50 \
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
--max_epoch=100 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=150 \
--beta1=0.5 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=50 \
\
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
--save_snapshot_epochs=25,50,75,100 \
--result_dir=$RESULT_DIR \
