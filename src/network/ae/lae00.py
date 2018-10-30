import math
import torch
import torch.nn as nn
from .__ae__ import Autoencoder, forward_types

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)

class Spatialize(nn.Module):
    def __init__(self, size):
        super(Spatialize, self).__init__()
        self.size = size

    def forward(self, x):
        nf = x.size(1) / (self.size * self.size)
        return torch.reshape(x, shape=(-1, nf, self.size, self.size))

class LadderAutoencoder00(Autoencoder):
    # architecture based on variational ladder autoencoder (not ladder variational autoencoder)

    def build(self):
        self.num_blocks = int(math.log2(self.args.img_size)) - 1

        self.z_sizes = list()
        z_remainder = self.args.z_size
        for i in range(self.num_blocks):
            if i == (self.num_blocks - 1):
                self.z_sizes.append(z_remainder)
            else:
                z_remainder = int(z_remainder / 2)
                self.z_sizes.append(z_remainder)
        self.z_sizes = [32, 32, 64, 64, 64]

        #       8   16   32   64   128   256   512
        nfs = (64, 128, 256, 512, 1024, 2048, 4096)
        nfs = (self.args.img_ch,) + nfs[:self.num_blocks-1] + (self.z_sizes[-1],)

        if self.args.img_size == 28:
            z_k_size = 7
        else:
            z_k_size = 4

        encoder1 = list()
        encoder2 = list()
        for i in range(self.num_blocks):
            f_size = int(self.args.img_size / (2 ** (i + 1)))
            nf = int(max(1, self.z_sizes[i] / (f_size * f_size)))

            if i == 0:
                encoder1.append(
                    nn.Sequential(
                        nn.Conv2d(nfs[i], nfs[i+1], 4, 2, 1, bias=True),
                        nn.LeakyReLU(0.2, inplace=True)))
                encoder2.append(
                    nn.Sequential(
                        nn.Conv2d(nfs[i+1], nf, 1, 1, 0, bias=True),
                        Flatten(),
                        nn.Conv2d(f_size * f_size * nf, self.z_sizes[i], 1, 1, 0, bias=True)))

            elif i == (self.num_blocks - 1):
                encoder1.append(nn.Conv2d(nfs[i], nfs[i+1], z_k_size, 1, 0, bias=True))

            else:
                encoder1.append(
                    nn.Sequential(
                        nn.Conv2d(nfs[i], nfs[i+1], 4, 2, 1, bias=True),
                        nn.BatchNorm2d(nfs[i+1], affine=True),
                        nn.LeakyReLU(0.2, inplace=True)))
                encoder2.append(
                    nn.Sequential(
                        nn.Conv2d(nfs[i+1], nf, 1, 1, 0, bias=True),
                        Flatten()))
        self.encoder = nn.ModuleList([nn.ModuleList(encoder1), nn.ModuleList(encoder2)])

        decoder1 = list()
        decoder2 = list()
        for i in range(self.num_blocks, 0, -1):
            f_size = int(self.args.img_size / (2 ** i))
            nf = int(max(1, self.z_sizes[i-1] / (f_size * f_size)))

            if i == 1:
                decoder2.append(
                    nn.Sequential(
                        nn.Conv2d(self.z_sizes[i-1], f_size * f_size * nf, 1, 1, 0, bias=True),
                        Spatialize(f_size),
                        nn.Conv2d(nf, nfs[i], 1, 1, 0, bias=True)))
                decoder1.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(nfs[i] * 2, nfs[i-1], 4, 2, 1, bias=True),
                        nn.Tanh()))

            elif i == self.num_blocks:
                decoder1.append(nn.ConvTranspose2d(nfs[i], nfs[i-1], z_k_size, 1, 0, bias=True))

            else:
                decoder2.append(
                    nn.Sequential(
                        nn.Conv2d(self.z_sizes[i-1], f_size * f_size * nf, 1, 1, 0, bias=True),
                        Spatialize(f_size),
                        nn.Conv2d(nf, nfs[i], 1, 1, 0, bias=True)))
                decoder1.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(nfs[i] * 2, nfs[i-1], 4, 2, 1, bias=True),
                        nn.BatchNorm2d(nfs[i-1], affine=True),
                        nn.LeakyReLU(0.2, inplace=True)))
        self.decoder = nn.ModuleList([nn.ModuleList(decoder1), nn.ModuleList(decoder2)])

        for m in encoder1 + encoder2 + decoder1 + decoder2:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                pass

    def forward(self, x, _x=None, forward_type=forward_types[0], dout=None):
        assert (self.encoder and self.decoder) is not None
        assert forward_type in forward_types

        # all
        if forward_type == forward_types[0]:
            z = self.encode(x)
            _x = self.decode(z)
            result = self.calc_loss(_x, x)

        # encoder
        elif forward_type == forward_types[1]:
            result = self.encode(x)

        # decoder
        elif forward_type == forward_types[2]:
            z = x
            result = self.decode(z)

        # autoencoder
        elif forward_type == forward_types[3]:
            z = self.encode(x)
            result = self.decode(z)

        # loss
        elif forward_type == forward_types[4]:
            result = self.calc_loss(_x, x)

        else:
            result = None
        return result

    def encode(self, x):
        z_list = list()
        h = x
        for i in range(self.num_blocks):
            h = self.encoder[0][i].forward(h)
            if i == (self.num_blocks - 1):
                z = h
            else:
                z = self.encoder[1][i].forward(h)
            z_list.append(z)
        z_list.reverse()
        z = torch.cat(z_list, dim=1)
        return z

    def decode(self, z):
        z_list = list()
        offset = 0
        for z_size in self.z_sizes:
            z_list.append(z[:, offset:(offset + z_size)])
            offset += z_size
        z_list.reverse()

        h = None
        for i in range(self.num_blocks):
            z = z_list[i]
            if h is None:
                x = z
            else:
                x = self.decoder[1][i-1].forward(z)
                x = torch.cat((h, x), dim=1)
            h = self.decoder[0][i].forward(x)
        y = h
        return y
