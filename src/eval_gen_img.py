import os
import numpy as np
import torch
import option
import kde

GEN_IMG_DIR='./result/evaluation/mnist/z_size_8/best_dataset=mnist_anneal=60000 _num_flow_layers=8_dimz=8/best_80'
DATASET_NAME='mnist2'
VAL_DATASET_PATH='./data/mnist2/valid'
EVAL_DATASET_PATH='./data/mnist2/test'
BATCH_SIZE=100

# GEN_IMG_DIR='./result/evaluation/tfd/z_size_32/tfd_num_flow_layers=8 _dimz=32/397_best'
# GEN_IMG_DIR='./result/evaluation/tfd/z_size_15/best_tfd_num_flow_layers=8_dimz=15/best_347'
# DATASET_NAME='tfd'
# VAL_DATASET_PATH='./data/tfd/inputs_valid.npy'
# EVAL_DATASET_PATH='./data/tfd/inputs_test.npy'
# BATCH_SIZE=10

IMG_SIZE=28
IMG_CH=1
SIGMA_RANGE=np.arange(0.1, 0.3, 0.01)
# MNIST: sigma_range=np.arange(0.1, 0.3, 0.01)
# TFD: sigma_range=np.arange(0.001, 0.200, 0.005)


class Args(object):
    def __init__(self):
        self.img_size = IMG_SIZE
        self.img_ch = IMG_CH
        self.batch_size = BATCH_SIZE
args = Args()


def main():
    dataset_cls = option.dataset_dict[DATASET_NAME]
    val_dataset = dataset_cls(args, VAL_DATASET_PATH, True)
    eval_dataset = dataset_cls(args, EVAL_DATASET_PATH, True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2)

    gen_img_paths = list()
    for img_name in os.listdir(GEN_IMG_DIR):
        img_path = os.path.join(GEN_IMG_DIR, img_name)
        # print(img_path)
        gen_img_paths.append(img_path)

    log_prob, std, sigma = kde.kde_eval(
        val_data_loader, gen_img_paths, BATCH_SIZE, sigma_range=SIGMA_RANGE, verbose=False)

    test_log_prob, test_std, _ = kde.kde_eval(
        eval_data_loader, gen_img_paths, BATCH_SIZE, sigma_range=[sigma])

    print('\tValidation: %.2f (%.2f), sigma: %.2f' % (log_prob, std, sigma))
    print('\tTest      : %.2f (%.2f)\n' % (test_log_prob, test_std))


if __name__ == "__main__":
    main()
