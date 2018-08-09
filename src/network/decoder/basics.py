import numpy as np
import torch.nn as nn

class BasicDecoder00(nn.Module):
    # basics decoder network

    def __init__(self, args):
        super(BasicDecoder00, self).__init__()
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch
        self.code_size = args.code_size

        num_blocks = int(np.log2(np.min((self.img_h, self.img_w)))) - 1
        nfs = (2048, 1024, 512, 256, 128)
        nfs = (self.code_size,) + nfs[len(nfs)-(num_blocks-1):] + (self.img_ch,)

        decoder = list()
        for i in range(num_blocks):
            if i == 0:
                decoder.append(nn.ConvTranspose2d(nfs[i], nfs[i+1], 4, 1, 0, bias=False))
            elif i == (num_blocks - 1):
                decoder.append(nn.ConvTranspose2d(nfs[i], nfs[i+1], 4, 2, 1, bias=False))
                decoder.append(nn.Tanh())
            else:
                decoder.append(nn.ConvTranspose2d(nfs[i], nfs[i+1], 4, 2, 1, bias=False))
                decoder.append(nn.BatchNorm2d(nfs[i+1], affine=True))
                decoder.append(nn.LeakyReLU(0.2, inplace=True))
        self.decoder = nn.Sequential(*decoder)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                pass

    def forward(self, *z):
        _x = self.decoder.forward(z[0])
        return _x