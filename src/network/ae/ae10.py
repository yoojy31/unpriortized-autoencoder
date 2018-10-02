import math
import torch.nn as nn
from .__ae__ import Autoencoder

class Autoencoder10(Autoencoder):
    "Autoencoder00 + Pixel Shuffle Upsampling"

    def build(self):
        num_blocks = int(math.log2(self.args.img_size)) - 1
        #       8   16   32   64   128   256   512
        nfs = (64, 128, 256, 512, 1024, 2048, 4096)
        nfs = (self.args.img_ch,) + nfs[:num_blocks-1] + (self.args.z_size,)

        if self.args.img_size == 28:
            z_k_size = 7
        else:
            z_k_size = 4

        encoder = list()
        for i in range(num_blocks):
            if i == 0:
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], 3, 2, 1, bias=True))
                encoder.append(nn.LeakyReLU(0.2, inplace=True))
            elif i == (num_blocks - 1):
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], z_k_size, 1, 0, bias=True))
            else:
                encoder.append(nn.Conv2d(nfs[i], nfs[i+1], 3, 2, 1, bias=True))
                encoder.append(nn.BatchNorm2d(nfs[i+1], affine=True))
                encoder.append(nn.LeakyReLU(0.2, inplace=True),)
        self.encoder = nn.Sequential(*encoder)

        decoder = list()
        for i in range(num_blocks, 0, -1):
            # print(nfs[i], nfs[i-1])
            if i == 1:
                decoder.append(nn.Conv2d(nfs[i], nfs[i-1] * 4, 3, 1, 1, bias=True))
                decoder.append(nn.PixelShuffle(2))
            elif i == num_blocks:
                decoder.append(nn.ConvTranspose2d(nfs[i], nfs[i-1], z_k_size, 1, 0, bias=True))
            else:
                decoder.append(nn.Conv2d(nfs[i], nfs[i-1] * 4, 3, 1, 1, bias=True))
                decoder.append(nn.PixelShuffle(2))
                decoder.append(nn.BatchNorm2d(nfs[i-1], affine=True))
                decoder.append(nn.LeakyReLU(0.2, inplace=True))
        self.decoder = nn.Sequential(*decoder)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                pass
