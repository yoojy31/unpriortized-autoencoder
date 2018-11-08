import os
import numpy as np
import scipy.misc
import torch
from .__dataset__ import ImgDataset

class Cifar(ImgDataset):
    def __init__(self, args, dataset_path, pre_processing=True):
        self.args = args
        self.dataset_root = dataset_path
        self.pre_processing = pre_processing
        self.img_names = os.listdir(dataset_path)
        self.num_imgs = len(self.img_names)
        self.label_mapping = {
            'cat':0, 'dog':1, 'airplane':2, 'ship':3, 'deer':4,
            'automobile':5, 'truck':6, 'frog':7, 'bird':8, 'horse':9}

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.img_names[idx])
        x = scipy.misc.imread(img_path, mode='RGB')
        # if len(x.shape) == 0:
        #     # print(img_path)
        # else:
        y = img_path.split('/')[4].split('_')[1].split('.')[0]
        return {'image': self.pre_process(x) if self.pre_processing else x, 'label': self.label_mapping[y]}

    def pre_process(self, x):
        if (self.args.img_size is not None) or \
            (x.shape[0] == self.args.img_size and x.shape[1] == self.args.img_size):
            # try:
            size = (self.args.img_size, self.args.img_size)
            x = scipy.misc.imresize(x, size, interp='bicubic')
            # except ValueError:
            #     print(x.shape)

        x = np.transpose(x, (2, 0, 1))
        x = (x / 127.5) - 1
        x = torch.from_numpy(x).float()
        return x

    def post_process(self, x):
        x = torch.clamp(x, min=-1, max=1)
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = (x + 1) * 127.5
        return x
