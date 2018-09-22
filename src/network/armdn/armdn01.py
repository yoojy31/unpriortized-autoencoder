import math
import torch.nn as nn
from .__armdn__ import ARMDN

class ARMDN01(ARMDN):
    def build(self):
        num_blocks = int(math.log2(self.args.z_size)) + 1
        #      1   2   4   8   16   32   64   128   256   512
        nfs = (2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        nfs = nfs[:num_blocks] + (self.args.n_gauss * 3,)

        armdn = list()
        for i in range(num_blocks):
            if i == (num_blocks - 1):
                armdn.append(nn.Conv1d(
                    nfs[i], nfs[i+1], kernel_size=1,
                    stride=1, dilation=1, bias=True))
            else:
                armdn.append(nn.Conv1d(
                    nfs[i], nfs[i+1], kernel_size=2,
                    stride=1, dilation=2**i, bias=True))
                armdn.append(nn.BatchNorm1d(nfs[i+1], affine=True))
                armdn.append(nn.LeakyReLU(0.2))
        self.armdn = nn.Sequential(*armdn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                pass
