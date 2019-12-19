import math
import torch
import torch.nn as nn
from .__armdn__ import ARMDN

class ARMDN00(ARMDN):
    def build(self):
        num_blocks = math.ceil(math.log2(self.args.z_size)) + 1
        #      1   2   4   8  16   32   64  128   256   512
        nfs = (2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        # nfs = (2,  8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        nfs = nfs[:num_blocks] + (self.args.n_gauss * 3,)

        armdn = list()
        for i in range(num_blocks):
            if i == (num_blocks - 1):
                armdn.append(nn.Conv1d(
                    nfs[i], nfs[i+1], kernel_size=1,
                    stride=1, dilation=1, bias=True))
            else:
                # armdn.append(nn.Conv1d(
                #     nfs[i], nfs[i], kernel_size=1,
                #     stride=1, bias=True))
                armdn.append(nn.Conv1d(
                    nfs[i], nfs[i+1], kernel_size=2,
                    stride=1, dilation=2**i, bias=True))
                # armdn.append(OptimizedDilatedConv1d(
                #     nfs[i], nfs[i+1], dilation=2**i))
                armdn.append(nn.BatchNorm1d(nfs[i+1]))
                armdn.append(nn.LeakyReLU(0.2))
        self.armdn = nn.Sequential(*armdn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.00, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                pass

class OptimizedDilatedConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(OptimizedDilatedConv1d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 1), stride=(1, 1))
        self.dilation = dilation

    def forward(self, x):
        shape = x.shape
        # print(x.shape)
        pad = (self.dilation - int(math.ceil(shape[2] % self.dilation)))
        x = nn.functional.pad(x, (0, pad))
        x = torch.reshape(x, (shape[0], shape[1], -1, self.dilation))
        # print(x.shape)
        y = self.conv.forward(x)
        shape = y.shape
        y = torch.reshape(y, (shape[0], shape[1], -1))
        if pad != 0:
            y = y[:, :, :-1 * pad]
        # print('')
        return y
