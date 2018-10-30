LOAD_SNAPSHOT_DIR=result/bsds/20181018-195432-armdn00/epoch-30
RESULT_DIR=bsds/armdn/`(date "+%Y%m%d-%H%M%S")`-ae00-mse1.0-1bnperc1.0

mkdir $RESULT_DIR
mkdir $RESULT_DIR/copy
cp -r src $RESULT_DIR/copy
cp eval_armdn.sh $RESULT_DIR/copy

python3 ./src/eval_armdn.py \
\
--devices=0 \
\
--armdn=00 \
--ae=ae00 \
--z_size=64 \
\
--n_gauss=20 \
--tau=1.0 \
\
--eval_dataset=arg-bsds \
--eval_set_path=./data/bsds/augmented/test \
\
--batch_size=128 \
--img_size=64 \
--img_ch=3 \
\
--load_snapshot_dir=$LOAD_SNAPSHOT_DIR \
--result_dir=$RESULT_DIR \
\




# eval_parser.add_argument('--devices', type=str, default='0')

# eval_parser.add_argument('--ae', type=str, default=None, help=network_dict.keys())
# eval_parser.add_argument('--armdn', type=str, default=None, help=network_dict.keys())
# eval_parser.add_argument('--z_size', type=int, default=64)

# train_parser.add_argument('--n_gauss', type=int, default=20)
# train_parser.add_argument('--tau', type=float, default=1.0)

# train_parser.add_argument('--eval_dataset', type=str, help=dataset_dict.keys())
# train_parser.add_argument('--eval_set_path', type=str)

# train_parser.add_argument('--batch_size', type=int, default=128)
# train_parser.add_argument('--img_size', type=int, default=64)
# train_parser.add_argument('--img_ch', type=int, default=3)

# train_parser.add_argument('--load_snapshot_dir', type=str)
# train_parser.add_argument('--result_dir', type=str)