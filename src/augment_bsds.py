import os
from tqdm import tqdm
import numpy as np
import scipy.misc
import torch
import utils
import option

def augment_bsds300():
    # [Neural Autoregressive Distribution Estimation 7.3.2]
    # 1.    We trained our models by using patches randomly drawn
    #       from 180 images in the training subset of BSDS300
    # 2.    We used the remaining 20 images in the training subset as validation data.
    #       We used 1000 random patches from the validation subset
    # 3.    We measured the performance of each model by their log-likelihood
    #       on one million patches drawn randomly from the test subset of 100 images
    #       not present in the training data.

    train_dir = os.path.join(args.result_dir, 'train')
    valid_dir = os.path.join(args.result_dir, 'valid')
    test_dir = os.path.join(args.result_dir, 'test')
    utils.make_dir(train_dir)
    utils.make_dir(valid_dir)
    utils.make_dir(test_dir)

    dataset_cls = option.dataset_dict[args.dataset]
    train_dataset = dataset_cls(args, args.train_set_path, False)
    test_dataset = dataset_cls(args, args.test_set_path, False)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2)

    train_ppi = int(2700000 / 180)
    valid_ppi = int(1000 / 20)
    test_ppi = int(1000000 / 100)

    train_loader_pbar = tqdm(train_data_loader)
    for i, train_batch in enumerate(train_loader_pbar):
        img = train_batch['image']
        img = np.squeeze(img.detach()).numpy()

        if i < 180:
            patches = train_dataset.augment(img, train_ppi)
            for j, patch in enumerate(patches):
                patch_path = os.path.join(train_dir, '%03d-%05d.npy' % (i, j))
                np.save(patch_path, patch)
            train_loader_pbar.set_description(
                '[augmentation] train:%d/180 valid:0/20 |' % (i + 1))

        else:
            patches = train_dataset.augment(img, valid_ppi)
            for j, patch in enumerate(patches):
                patch_path = os.path.join(valid_dir, '%03d-%05d.npy' % (i, j))
                np.save(patch_path, patch)
            train_loader_pbar.set_description(
                '[augmentation] train:180/180 valid:%d/20 |' % (i - 179))

    test_loader_pbar = tqdm(test_data_loader)
    for i, test_batch in enumerate(test_loader_pbar):
        img = test_batch['image']
        img = np.squeeze(img.detach()).numpy()
        patches = test_dataset.augment(img, test_ppi)

        for j, patch in enumerate(patches):
            patch_path = os.path.join(test_dir, '%03d-%05d.npy' % (i, j))
            np.save(patch_path, patch)
            test_loader_pbar.set_description(
                '[augmentation] test:%d/100 |' % (i + 1))

if __name__ == '__main__':
    args = option.preprocess_parser.parse_args()

    if args.dataset == 'bsds':
        augment_bsds300()
