import numpy as np
import torch.nn as nn

class BasicsEncoder00(nn.Module):
    # bagics encoder network

    def __init__(self, args):
        super(BasicsEncoder00, self).__init__()
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch
        self.code_size = args.code_size

        num_blocks = int(np.log2(np.min((self.img_h, self.img_w)))) - 1
        nfs = (128, 256, 512, 1024, 2048)
        nfs = (self.img_ch,) + nfs[:num_blocks-1] + (self.code_size,)

        encoder = list()
        for i in range(num_blocks):
            if i == 0:
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], 4, 2, 1, bias=False))
                encoder.append(nn.LeakyReLU(0.2, inplace=True))
            elif i == (num_blocks - 1):
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], 4, 1, 0, bias=False))
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
            else:
                pass

    def forward(self, *x):
        z = self.encoder.forward(x[0])
        return z
