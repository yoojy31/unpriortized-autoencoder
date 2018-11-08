#!/bin/bash

LOAD_SNAPSHOT_DIR='result/training/celeba/64x64/100/ae/20181105-182946-ae00(mse1.0-perc0.1)-(mwup0)/snapshot/epoch-150'
RESULT_DIR=./result/training/celeba/64x64/100/armdn/`(date "+%Y%m%d-%H%M%S")`-'armdn00(20)-ae00(mse1.0-perc0.1-mwup0)'

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_armdn.sh $RESULT_DIR/copy

python3 ./src/train_armdn.py \
--devices=1 \
\
--ae=ae00 \
--armdn=armdn00 \
--z_size=100 \
--static_z_size=100 \
--z_dout_rate=0.0 \
\
--n_gauss=20 \
--tau=1.0 \
\
--train_dataset=celeb \
--valid_dataset=celeb \
--train_set_path=./data/celeba/align-crop-128/train-valid \
--valid_set_path=./data/celeba/align-crop-128/test \
\
--batch_size=64 \
--img_size=64 \
--img_ch=3 \
\
--init_epoch=0 \
--max_epoch=90 \
--lr=2e-4 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=15,35,60 \
--beta1=0.5 \
\
--perc_w=0.0 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=50 \
\
--save_snapshot_epochs=20,40,50,60,70,80,90 \
--result_dir=$RESULT_DIR \
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
# \
# --ordering
# --dataset=celeb \
# --train_set_path=./data/celeba/align-crop-128/train-valid \
# --valid_set_path=./data/celeba/align-crop-128/test \