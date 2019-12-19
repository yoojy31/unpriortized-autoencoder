import os
import numpy as np
import scipy.misc
import torch
from .__dataset__ import ImgDataset
from .__tools__ import cvt_rgb2gray, sample_patches

class BSDS(ImgDataset):
    def __init__(self, args, dataset_path, pre_processing=None):
        self.args = args
        self.dataset_root = dataset_path
        self.pre_processing = pre_processing
        self.img_names = os.listdir(dataset_path)
        self.num_imgs = len(self.img_names)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.img_names[idx])
        x = scipy.misc.imread(img_path, mode='RGB')
        return {'image': self.pre_process(x) if self.pre_processing else x}

    def pre_process(self, x):
        x = self.augment(x, 1)[0]
        x = torch.from_numpy(x).float()
        x = torch.unsqueeze(x, dim=1)
        x = torch.unsqueeze(x, dim=2)
        return x

    def post_process(self, x):
        x = torch.squeeze(x)
        x = x.detach().numpy()
        return x

    def augment(self, *x):
        # [Neural Autoregressive Distribution Estimation 7.3.2]
        # 1.    we use 8-by-8-pixel patches of monochrome natural images
        #       Pixels in this dataset can take a finite number of
        #       brightness values ranging from 0 to 255.
        # 2.    We added uniformly distributed noise between 0 and 1
        #       to the brightness of each pixel. We then divided by 256,
        #       making the pixels take continuous values in the range [0, 1].
        # 3.    We subtracted the mean pixel value from each patch.
        # 4.    The average intensity of each patch was subtracted from
        #       each pixel’s value. After this, all datapoints lay on a 63-
        #       dimensional subspace, for this reason only 63 pixels were
        #       modelled, discarding the bottom-right pixel.
        # 5.    All of the results in this section were obtained by fitting
        #       the pixels in a raster-scan order.
        img = x[0]
        ppi = x[1]

        img = cvt_rgb2gray(img)
        patches = sample_patches(img, ppi, 8)
        for i, patch in enumerate(patches):
            noise = np.random.uniform(low=0.0, high=1.0, size=patch.shape)
            patch = (patch.astype(np.float) + noise) / 256
            patch = patch - np.mean(patch)
            patches[i] = np.reshape(patch, newshape=(64))[:63]
        return patches

class AugmentedBSDS(ImgDataset):
    def __init__(self, args, dataset_path, pre_processing=None):
        self.args = args
        self.dataset_root = dataset_path
        self.pre_processing = pre_processing
        self.data_names = os.listdir(dataset_path)
        self.num_data = len(self.data_names)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.data_names[idx])
        x = np.load(img_path)
        if self.pre_processing:
            x = self.pre_process(x)
        return {'image': x}

    def pre_process(self, x):
        x = torch.from_numpy(x).float()
        x = torch.unsqueeze(x, dim=1)
        x = torch.unsqueeze(x, dim=2)
        return x

    def post_process(self, x):
        x = x.detach().numpy()
        return x
