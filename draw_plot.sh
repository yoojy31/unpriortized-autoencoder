# LOAD_SNAPSHOT_DIR='result/training/mnist/28x28/8/ae/20181107-202409-ae00(mse1.0)-(mwup0)/snapshot/epoch-80'
# RESULT_DIR=./result/plot/mnist/28x28/8/`(date "+%Y%m%d-%H%M%S")`-'ae00(mse1.0)-(mwup0)'

# LOAD_SNAPSHOT_DIR='./result/training/mnist/28x28/8/ae/20181107-202724-ae00(mse1.0)-(mwup16)/snapshot/epoch-80'
# RESULT_DIR=./result/plot/mnist/28x28/8/`(date "+%Y%m%d-%H%M%S")`-'ae00(mse1.0)-(mwup16)'

# LOAD_SNAPSHOT_DIR='./result/training/mnist/28x28/8/ae/20181107-214211-ae00(mse1.0)-(mwup32)/snapshot/epoch-80'
# RESULT_DIR=./result/plot/mnist/28x28/8/`(date "+%Y%m%d-%H%M%S")`-'ae00(mse1.0)-(mwup32)'

# LOAD_SNAPSHOT_DIR='./result/training/mnist/28x28/8/ae/20181107-215007-ae00(mse1.0)-(mwup48)/snapshot/epoch-80'
# RESULT_DIR=./result/plot/mnist/28x28/8/`(date "+%Y%m%d-%H%M%S")`-'ae00(mse1.0)-(mwup48)'

# LOAD_SNAPSHOT_DIR='./result/training/mnist/28x28/2/ae/20181107-220350-ae00(mse1.0)-(mwup0)/snapshot/epoch-80'
# RESULT_DIR=./result/plot/mnist/28x28/2/`(date "+%Y%m%d-%H%M%S")`-'ae00(mse1.0)-(mwup0)'

LOAD_SNAPSHOT_DIR='./result/training/mnist/28x28/2/ae/20181107-220416-ae00(mse1.0)-(mwup20)/snapshot/epoch-50'
RESULT_DIR=./result/plot/mnist/28x28/2/`(date "+%Y%m%d-%H%M%S")`-'ae00(mse1.0)-(mwup20)'

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp draw_plot.sh $RESULT_DIR/copy

python3 ./src/draw_plot.py \
--ae=ae00 \
--z_size=2 \
\
--dataset=mnist \
--dataset_path=./data/mnist/t10k-images.idx3-ubyte \
--batch_size=64 \
--max_iters=100 \
\
--img_size=28 \
--img_ch=1 \
\
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
--result_dir=$RESULT_DIR \
