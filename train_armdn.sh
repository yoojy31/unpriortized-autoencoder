#!/bin/bash

LOAD_SNAPSHOT_DIR='snapshot-dir-path-of-trained-autoencoder'
RESULT_DIR='result-dir-path'

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_armdn.sh $RESULT_DIR/copy

python3 ./src/train_armdn.py \
--devices=1 \
\
--ae=ae00 \
--armdn=armdn00 \
--z_size=200 \
--static_z_size=200 \
--z_dout_rate=0.0 \
\
--n_gauss=30 \
--tau=1.0 \
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
\
--init_epoch=0 \
--max_epoch=200 \
--lr=2e-4 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=15,35,60,100 \
--beta1=0.5 \
\
--perc_w=0.0 \
\
--valid_iter_intv=50 \
\
--save_snapshot_epochs=50,90,120,150,200 \
--result_dir=$RESULT_DIR \
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \

