#!/bin/bash

# LOAD_SNAPSHOT_DIR='result/training/celeba/64x64/100/ae/20181105-182946-ae00(mse1.0-perc0.1)-(mwup0)/snapshot/epoch-150'
# LOAD_SNAPSHOT_DIR='result/training/mnist/28x28/8/ae/20181107-202724-ae00(mse1.0)-(mwup16)/snapshot/epoch-80'
# LOAD_SNAPSHOT_DIR='result/training/celeba/64x64/100/ae/20181105-182830-ae00(mse1.0-perc0.1)-(mwup100)/snapshot/epoch-150'
# RESULT_DIR=./result/training/celeba/64x64/100/armdn/`(date "+%Y%m%d-%H%M%S")`-'gmmn-ae00(mse1.0-perc0.1-mwup100)'
RESULT_DIR=./result/training/bsds/`(date "+%Y%m%d-%H%M%S")`-'armdn00(5)'

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_armdn.sh $RESULT_DIR/copy

python3 ./src/train_armdn.py \
--devices=1 \
\
--ae=none \
--armdn=armdn00 \
--z_size=64 \
--static_z_size=64 \
--z_dout_rate=0.0 \
\
--n_gauss=5 \
--tau=1.0 \
\
--train_dataset=aug-bsds \
--valid_dataset=aug-bsds \
--train_set_path=./data/bsds300/augmented/train \
--valid_set_path=./data/bsds300/augmented/test \
\
--batch_size=128 \
--img_size=64 \
--img_ch=3 \
\
--init_epoch=0 \
--max_epoch=150 \
--lr=2e-4 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=15,35,60,100 \
--beta1=0.5 \
\
--perc_w=0.0 \
\
--eval_epoch_intv=10 \
--valid_iter_intv=50 \
\
--save_snapshot_epochs=20,40,50,60,70,80,90 \
--result_dir=$RESULT_DIR \
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
# \
# --train_dataset=celeb \
# --valid_dataset=celeb \
# --train_set_path=./data/celeba/align-crop-128/train-valid \
# --valid_set_path=./data/celeba/align-crop-128/test \
#
# --train_dataset=mnist \
# --valid_dataset=mnist \
# --train_set_path=./data/mnist/train-images.idx3-ubyte \
# --valid_set_path=./data/mnist/t10k-images.idx3-ubyte \
#
# --train_dataset=aug-bsds \
# --valid_dataset=aug-bsds \
# --train_set_path=./data/bsds300/augmented/train \
# --valid_set_path=./data/bsds300/augmented/valid \
