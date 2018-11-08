LOAD_SNAPSHOT_DIR='./result/training/celeba-shoes/64x64/100/ae/20181106-192700-ae00(mse1.0-perc0.1)-(mwup100)/snapshot/epoch-150'
RESULT_DIR=./result/interpolation/celeba-shoes/64x64/100/`(date "+%Y%m%d-%H%M%S")`-'ae00-ae00(mse1.0-perc0.1)-(mwup100)'

# LOAD_SNAPSHOT_DIR='./result/training/celeba-shoes/64x64/100/ae/20181106-192719-ae00(mse1.0-perc0.1)-(mwup0)/snapshot/epoch-150'
# RESULT_DIR=./result/interpolation/celeba-shoes/64x64/100/`(date "+%Y%m%d-%H%M%S")`-'ae00-ae00(mse1.0-perc0.1)-(mwup0)'

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp interpolate_z.sh $RESULT_DIR/copy

python3 ./src/interpolate_z.py \
--ae=ae00 \
--z_size=100 \
\
--dataset=two \
--dataset_path=./data/celeba-shoes/test \
--img_size=64 \
--img_ch=3 \
\
--batch_size=192 \
--pairs=14,146\|45,11\|30,33\|53,84 \
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
--result_dir=$RESULT_DIR \
