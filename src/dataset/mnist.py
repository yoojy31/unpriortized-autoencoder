from struct import unpack
import numpy as np
import scipy.misc
import torch
from .__dataset__ import ImgDataset

class MNIST(ImgDataset):

    def __init__(self, args, dataset_path, pre_processing=True):
        self.args = args
        self.dataset_root = dataset_path
        self.pre_processing = pre_processing

        img_file_path = dataset_path
        lab_file_path = dataset_path.replace(
            'images.idx3', 'labels.idx1')

        def load_dataset(img_file_path, lab_file_path):
            img_list = list()
            lab_list = list()

            imgs = open(img_file_path, 'rb')
            labs = open(lab_file_path, 'rb')
            _, __ = imgs.read(16), labs.read(8)

            while True:
                img = imgs.read(784)
                if not img:
                    break
                img = unpack(len(img) * 'B', img)
                img = np.reshape(img, (28, 28))
                img_list.append(img)

                lab = labs.read(1)
                lab = int(unpack(len(lab) * 'B', lab)[0])
                lab_list.append(lab)

            imgs.close()
            labs.close()
            return img_list, lab_list

        self.img_list, self.lab_list = \
            load_dataset(img_file_path, lab_file_path)
        self.num_imgs = len(self.img_list)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        x = self.img_list[idx]
        x = self.pre_process(x) if self.pre_processing else x
        y = self.lab_list[idx]
        # s = torch.zeros(self.code_size, 1, 1)
        return {'x': x, 'y': y} #, 's': s}

    def pre_process(self, x):
        if self.args.img_size is not None:
            x = scipy.misc.imresize(
                x, self.args.img_size,
                interp='bicubic')

        x = (x / 127.5) - 1
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()
        return x

    def post_process(self, x):
        x = torch.clamp(x, min=-1, max=1)
        x = x.detach().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = np.squeeze(x)
        x = (x + 1) * 127.5
        return x
