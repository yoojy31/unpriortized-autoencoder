#!/bin/bash

# LOAD_SNAPSHOT_DIR='result/training/mnist/28x28/8/ae/20181114-141604-ae02(mse1.0)-(mwup0)/snapshot/epoch-280'
LOAD_SNAPSHOT_DIR='./result/training/mnist/28x28/8/ae/20181115-040410-ae02(mse1.0)-(mwup96)/snapshot/epoch-390'
RESULT_DIR=./result/training/mnist/28x28/8/armdn/`(date "+%Y%m%d-%H%M%S")`-'armdn00(30)-ae02(mse1.0-mwup96)'

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp train_armdn.sh $RESULT_DIR/copy

python3 ./src/train_armdn.py \
--devices=0 \
\
--ae=ae02 \
--armdn=armdn00 \
--z_size=8 \
--static_z_size=8 \
--z_dout_rate=0.0 \
\
--n_gauss=30 \
--tau=1.0 \
\
--train_dataset=mnist2 \
--valid_dataset=mnist2 \
--test_dataset=mnist2 \
--train_set_path=./data/mnist2/train \
--valid_set_path=./data/mnist2/valid \
--test_set_path=./data/mnist2/test \
\
--batch_size=128 \
--img_size=28 \
--img_ch=1 \
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
--eval_epoch_intv=10 \
--valid_iter_intv=50 \
\
--save_snapshot_epochs=50,90,120,150 \
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
