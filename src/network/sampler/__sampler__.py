import numpy as np
import torch
import torch.nn as nn

class Sampler(nn.Module):
    def __init__(self, args):
        super(Sampler, self).__init__()
        self.code_size = args.code_size
        self.num_bin = args.num_bin

    def forward(self, *x):
        pass

    def sample_z_1(self, p):
        assert isinstance(p, torch.Tensor)
        p = p.cpu().detach().numpy()

        batch_size = p.shape[0]
        num_bin = p.shape[1]
        bin_size = 2.0 / num_bin

        s = list()
        for i in range(batch_size):
            y = np.random.choice(num_bin, p=p[i])
            sample = y * bin_size + np.random.uniform(0, bin_size) - 1
            s.append(np.reshape(sample, newshape=(1, 1)))

        s = np.concatenate(s, axis=0)
        s = np.reshape(s, newshape=(-1, 1, 1, 1))
        s = torch.from_numpy(s).float()
        s = s.cuda()
        return s

    def sample_z_2(self):
        # https://stackoverflow.com/questions/39836779/sampling-from-a-computed-multivariate-kernel-density-estimation
        return 0
