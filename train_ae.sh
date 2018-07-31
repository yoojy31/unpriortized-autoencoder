#!/bin/bash

RESULT_DIR=./result/celeb/basics_ae0/bagics00-bagics00-`date "+%02Y%m%d%H%M%S"`

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train.sh $RESULT_DIR/copy

python3 ./src/train_ae.py \
--cur_device=0 \
\
--encoder=bagics00 \
--decoder=bagics00 \
--discrim=none \
\
--dataset=celeb \
--train_set_path=./dataset/celeba-crop-64/train \
--valid_set_path=./dataset/celeba-crop-64/test \
\
--trainer=basics_ae0 \
--start_epoch=0 \
--finish_epoch=30 \
--lr=2e-4 \
--beta1=0.9 \
\
--batch_size=128 \
--img_h=32 \
--img_w=32 \
--img_ch=3 \
--code_size=100 \
--num_bin=100 \
\
--print_intv=10 \
--valid_intv=10 \
\
--result_dir=$RESULT_DIR \
--snapshot_intv=10 \
# --snapshot_dir=$SNAPSHOT_DIR \
