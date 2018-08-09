#!/bin/bash
# test

RESULT_DIR=./result/celeb-32x32/lae-64/size-test-`date "+%Y%m%d-%H%M%S"`

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_ae.sh $RESULT_DIR/copy

python3 ./src/train_ae.py \
--cur_device=0 \
\
--encoder=basics00 \
--decoder=basics00 \
--discrim=none \
\
--dataset=celeb \
--train_set_path=./data/celeba-crop-64/train \
--valid_set_path=./data/celeba-crop-64/test \
\
--trainer=basics_ae0 \
--start_epoch=0 \
--finish_epoch=70 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_intv=20 \
--beta1=0.9 \
\
--batch_size=128 \
--img_size=28 \
--img_ch=3 \
--code_size=64 \
--num_bin=100 \
\
--print_intv=50 \
--valid_intv=50 \
\
--result_dir=$RESULT_DIR \
--snapshot_intv=20 \
# --snapshot_dir=$SNAPSHOT_DIR \
