import os
import numpy as np
import scipy.misc
import torch
from .__dataset__ import ImgDataset

class Celeb(ImgDataset):

    def __init__(self, args, dataset_path):
        self.img_size = (args.img_size, args.img_size)
        self.code_size = args.code_size
        self.dataset_root = dataset_path
        self.img_names = os.listdir(dataset_path)
        self.num_imgs = len(self.img_names)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.img_names[idx])
        x = scipy.misc.imread(img_path, mode='RGB')
        s = torch.zeros(self.code_size, 1, 1)
        return {'x': self.preprocessing(x), 's': s}

    def preprocessing(self, x):
        if self.img_size is not None:
            x = scipy.misc.imresize(
                x, self.img_size,
                interp='bicubic')

        x = np.transpose(x, (2, 0, 1))
        x = (x / 127.5) - 1
        x = torch.from_numpy(x).float()
        return x

    def inv_preprocessing(self, x):
        x = torch.clamp(x, min=-1, max=1)
        x = x.cpu().detach().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = (x + 1) * 127.5
        return x
