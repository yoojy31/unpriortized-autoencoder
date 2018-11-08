import os
import numpy as np
import scipy.misc
import torch
from .__dataset__ import ImgDataset

class Two(ImgDataset):
    def __init__(self, args, dataset_path, pre_processing=True):
        self.args = args
        self.dataset_root1 = dataset_path + '1'
        self.dataset_root2 = dataset_path + '2'
        self.pre_processing = pre_processing

        img_names1 = os.listdir(self.dataset_root1)
        img_names2 = os.listdir(self.dataset_root2)
        num_imgs1 = len(img_names1)
        num_imgs2 = len(img_names2)
        self.img_names = list()
        self.dataset_ids = list()
        for i in range(max(num_imgs1, num_imgs2)):
            if i < num_imgs1:
                self.img_names.append(img_names1[i])
                self.dataset_ids.append(0)
            if i < num_imgs2:
                self.img_names.append(img_names2[i])
                self.dataset_ids.append(1)
        self.num_imgs = len(self.img_names)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        if self.dataset_ids[idx] == 0:
            img_path = os.path.join(self.dataset_root1, self.img_names[idx])
        elif self.dataset_ids[idx] == 1:
            img_path = os.path.join(self.dataset_root2, self.img_names[idx])
        else:
            img_path = None
            assert self.dataset_ids[idx] in (1, 2)
        x = scipy.misc.imread(img_path, mode='RGB')
        return {'image': self.pre_process(x) if self.pre_processing else x,
                'label': self.dataset_ids[idx]}

    def pre_process(self, x):
        if self.args.img_size is not None:
            size = (self.args.img_size, self.args.img_size)
            x = scipy.misc.imresize(x, size, interp='bicubic')

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
