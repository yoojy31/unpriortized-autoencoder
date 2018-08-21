import numpy as np
import torch
import torch.nn as nn

class BasicEncoder00(nn.Module):
    # Basic encoder network

    def __init__(self, args):
        super(BasicEncoder00, self).__init__()
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.code_size = args.code_size

        num_blocks = int(np.log2(self.img_size)) - 1
        nfs = (128, 256, 512, 1024, 2048)
        nfs = (self.img_ch,) + nfs[:num_blocks-1] + (self.code_size,)

        if self.img_size == 28:
            last_k = 7
        else:
            # int(np.log2(self.img_size))
            # == np.log2(self.img_size)
            last_k = 4

        encoder = list()
        for i in range(num_blocks):
            if i == 0:
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], 4, 2, 1, bias=False))
                encoder.append(nn.LeakyReLU(0.2, inplace=True))
            elif i == (num_blocks - 1):
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], last_k, 1, 0, bias=False))
            else:
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], 4, 2, 1, bias=False))
                encoder.append(nn.BatchNorm2d(nfs[i+1], affine=True))
                encoder.append(nn.LeakyReLU(0.2, inplace=True),)
        self.encoder = nn.Sequential(*encoder)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        z = self.encoder.forward(x[0])
        return z, z

class LimitedEncoder00(BasicEncoder00):
    # Basic encoder network + ramp

    def forward(self, *x):
        z1 = self.encoder.forward(x[0])
        z2 = torch.clamp(z1, min=-1, max=1)
        return z1, z2

class LimitedEncoder01(BasicEncoder00):
    # Basic encoder network + tanh

    def forward(self, *x):
        z1 = self.encoder.forward(x[0])
        z2 = nn.functional.tanh(z1)
        return z1, z2
