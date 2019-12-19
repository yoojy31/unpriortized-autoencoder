#!/bin/bash

LOAD_SNAPSHOT_DIR='./result/training/celeba-shoes/64x64/100/armdn/20181120-225148-armdn00(30)-ae00(mse1.0-perc0.1)-(mwup0)/snapshot/epoch-200'
# LOAD_SNAPSHOT_DIR='./result/training/celeba-shoes/64x64/100/armdn/20181120-225200-armdn00(30)-ae00(mse1.0-perc0.1)-(mwup100)/snapshot/epoch-200'

# LOAD_SNAPSHOT_DIR='./result/training/celeba-shoes/64x64/100/ae/20181106-192719-ae00(mse1.0-perc0.1)-(mwup0)/snapshot/epoch-150'
# LOAD_SNAPSHOT_DIR='./result/training/celeba-shoes/64x64/100/ae/20181106-192700-ae00(mse1.0-perc0.1)-(mwup100)/snapshot/epoch-150'

# LOAD_SNAPSHOT_DIR='./result/training/cifar10/32x32/256/ae/20181119-161248-ae00(mse1.0)-(mwup0)/snapshot/epoch-100'
# LOAD_SNAPSHOT_DIR='./result/training/cifar10/32x32/256/ae/20181119-200502-ae00(mse1.0)-(mwup128)/snapshot/epoch-250'
# LOAD_SNAPSHOT_DIR='./result/training/celeba/128x128/200/ae/20181110-133815-ae00(mse1.0-perc0.1)-(mwup0)/snapshot/epoch-150'
# LOAD_SNAPSHOT_DIR='result/training/mnist/28x28/32/ae/20181111-233755-ae02(mse1.0)-(mwup0)/snapshot/epoch-70'
# LOAD_SNAPSHOT_DIR='result/training/mnist/28x28/32/ae/20181112-004319-ae02(mse1.0)-(mwup96)/snapshot/epoch-250'
# LOAD_SNAPSHOT_DIR='result/training/celeba/64x64/100/ae/20181105-182830-ae00(mse1.0-perc0.1)-(mwup100)/snapshot/epoch-150'
# RESULT_DIR=./result/training/cifar10/32x32/256/armdn/`(date "+%Y%m%d-%H%M%S")`-'armdn00(50)-ae00(mse1.0-mwup128)'

RESULT_DIR=./result/training/celeba-shoes/64x64/100/armdn/`(date "+%Y%m%d-%H%M%S")`-'armdn00(30)-ae00(mse1.0-perc0.1)-(mwup0)'
# RESULT_DIR=./result/training/celeba-shoes/64x64/100/armdn/`(date "+%Y%m%d-%H%M%S")`-'armdn00(30)-ae00(mse1.0-perc0.1)-(mwup100)'

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
--n_gauss=30 \
--tau=1.0 \
\
--train_dataset=two \
--valid_dataset=two \
--test_dataset=two \
--train_set_path=./data/celeba-shoes/train \
--valid_set_path=./data/celeba-shoes/test_subset \
--test_set_path=./data/celeba-shoes/test_subset \
\
--batch_size=128 \
--img_size=64 \
--img_ch=3 \
\
--init_epoch=200 \
--max_epoch=201 \
--lr=2e-4 \
--lr_decay_rate=5e-1 \
--lr_decay_epochs=15,35,60,100 \
--beta1=0.5 \
\
--perc_w=0.0 \
\
--eval_epoch_intv=5 \
--valid_iter_intv=50 \
\
--save_snapshot_epochs=50,90,120,150,200 \
--result_dir=$RESULT_DIR \
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \

# --train_dataset=celeb \
# --valid_dataset=celeb \
# --test_dataset=celeb \
# --train_set_path=./data/celeba/align-crop-128/train-valid \
# --valid_set_path=./data/celeba/align-crop-128/test \
# --test_set_path=./data/celeba/align-crop-128/test \
#
# --train_dataset=mnist \
# --valid_dataset=mnist \
# --train_set_path=./data/mnist/train-images.idx3-ubyte \
# --valid_set_path=./data/mnist/t10k-images.idx3-ubyte \
#
# --train_dataset=mnist2 \
# --valid_dataset=mnist2 \
# --test_dataset=mnist2 \
# --train_set_path=./data/mnist2/train \
# --valid_set_path=./data/mnist2/valid \
# --test_set_path=./data/mnist2/test \
#
# --train_dataset=aug-bsds \
# --valid_dataset=aug-bsds \
# --train_set_path=./data/bsds300/augmented/train \
# --valid_set_path=./data/bsds300/augmented/valid \
#
# --train_dataset=cifar \
# --valid_dataset=cifar \
# --test_dataset=cifar \
# --train_set_path=./data/cifar10/train \
# --valid_set_path=./data/cifar10/test \
# --test_set_path=./data/cifar10/test \
#
# --train_dataset=two \
# --valid_dataset=two \
# --test_dataset=two \
# --train_set_path=./data/celeba-shoes/train \
# --valid_set_path=./data/celeba-shoes/test_subset \
# --test_set_path=./data/celeba-shoes/test_subset \
