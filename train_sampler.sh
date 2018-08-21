#!/bin/bash

RESULT_DIR=./result/celeba-32x32/32/sampler/hier/hier11-larlae-`date "+%Y%m%d-%H%M%S"`
SNAPSHOT_DIR=./result/celeba-32x32/32/larlae/larlae-ar-satlins-all-20180820-225526/snapshot/epoch-120

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_sampler.sh $RESULT_DIR/copy

python3 ./src/train_sampler.py \
--cur_device=1 \
\
--encoder=latent_ar00 \
--decoder=basics00 \
--sampler=hier00 \
--discrim=none \
\
--dataset=celeb \
--train_set_path=./data/celeba/align-crop-128/train-valid \
--valid_set_path=./data/celeba/align-crop-128/test \
\
--trainer=basics_sampler0 \
--start_epoch=0 \
--finish_epoch=50 \
--lr=1e-3 \
--lr_decay_rate=5e-1 \
--lr_decay_intv=10 \
--beta1=0.9 \
\
--batch_size=128 \
--img_size=32 \
--img_ch=3 \
--code_size=32 \
--num_bin=50 \
\
--print_intv=50 \
--valid_intv=50 \
\
--result_dir=$RESULT_DIR \
--snapshot_dir=$SNAPSHOT_DIR \
--snapshot_intv=10 \

