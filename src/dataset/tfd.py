import numpy as np
import torch
from .__dataset__ import ImgDataset

class TFD(ImgDataset):
    def __init__(self, args, dataset_path, pre_processing=True):
        self.args = args
        self.dataset_root = dataset_path
        self.pre_processing = pre_processing
        self.data = np.load(dataset_path)
        self.num_imgs = self.data.shape[0]

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        x = self.data[idx]
        return {'image': self.pre_process(x) if self.pre_processing else x}

    def pre_process(self, x):
        x = np.reshape(x, (1, 48, 48))
        x = x / 255
        # x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()
        return x

    def post_process(self, x):
        x = torch.clamp(x, min=0, max=1)
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
        x = np.squeeze(x)
        x = x * 255
        x = np.reshape(x, (48, 48))
        return x
