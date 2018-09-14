#!/bin/bash

RESULT_DIR=./result/celeba/32x32/32/autoencoder/`(date "+%Y%m%d-%H%M%S")`-autoencoder00
# SNAPSHOT_DIR=result/celeba-32x32/32/lae/lae-satlins-0.0010-20180812-162033/snapshot/epoch-120

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_ae.sh $RESULT_DIR/copy

python3 ./src/train_ae.py \
--cur_device=0 \
\
--autoencoder=autoencoder00 \
--z_size=32 \
\
--dataset=celeb \
--train_set_path=./data/celeba/align-crop-128/train-valid \
--valid_set_path=./data/celeba/align-crop-128/test \
\
--batch_size=128 \
--img_size=32 \
--img_ch=3 \
\
--init_epoch=0 \
--max_epoch=150 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=20,50,80 \
--beta1=0.5 \
\
--eval_epoch_intv=10 \
--valid_iter_intv=50 \
\
--result_dir=$RESULT_DIR \
--save_snapshot_epochs=100, 130, 150 \
# --load_snapshot_dir=$SNAPSHOT_DIR \
