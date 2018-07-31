import numpy as np
import torch
import torch.nn as nn

class ARMLPEncoder00(nn.Module):
    # mlp based auto-regressive encoder

    def __init__(self, args):
        super(ARMLPEncoder00, self).__init__()
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch
        self.code_size = args.code_size

        num_blocks = int(np.log2(np.min((self.img_h, self.img_w)))) - 1
        nfs = (128, 256, 512, 1024, 2048)
        nfs = (self.img_ch,) + nfs[:num_blocks-1] + (self.code_size,)

        encoder1 = list()
        for i in range(num_blocks):
            if i == 0:
                encoder1.append(nn.Conv2d(nfs[i], nfs[i+1], 4, 2, 1, bias=False))
                encoder1.append(nn.LeakyReLU(0.2, inplace=True))
            elif i == (num_blocks - 1):
                encoder1.append(nn.Conv2d(nfs[i], nfs[i+1], 4, 1, 0, bias=False))
            else:
                encoder1.append(nn.Conv2d(nfs[i], nfs[i+1], 4, 2, 1, bias=False))
                encoder1.append(nn.BatchNorm2d(nfs[i+1], affine=True))
                encoder1.append(nn.LeakyReLU(0.2, inplace=True),)
        self.encoder1 = nn.Sequential(*encoder1)

        self.encoder2 = nn.Sequential(
            nn.Linear(args.code_size * 3, args.code_size * 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(args.code_size * 3, args.code_size * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(args.code_size * 2, args.code_size * 2, bias=False),
            nn.BatchNorm1d(args.code_size * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(args.code_size * 2, args.code_size, bias=False),
            nn.BatchNorm1d(args.code_size, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(args.code_size, args.code_size, bias=False),
            nn.BatchNorm1d(args.code_size, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(args.code_size, 1, bias=False))

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
        h = self.encoder1.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z = torch.zeros((batch_size, self.code_size))
        if h.is_cuda:
            z = z.cuda()

        for i in range(self.args.code_size):
            i = torch.zeros((batch_size, self.code_size))
            if h.is_cuda:
                i = i.cuda()
            i[:, i] = 1

            h_i = torch.cat((h, z.detach(), i), dim=1)
            z_i = torch.clamp(self.encoder2(h_i), min=-1.0, max=1.0)
            z[:, i:i+1] += z_i

        z = torch.view(batch_size, self.code_size, 1, 1)
        return z


class ARMLPEncoder01(ARMLPEncoder00):
    def forward(self, *x):
        h = self.encoder1.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z = torch.zeros((batch_size, self.code_size))
        if h.is_cuda:
            z = z.cuda()

        for i in range(self.args.code_size):
            i = torch.zeros((batch_size, self.code_size))
            if h.is_cuda:
                i = i.cuda()
            i[:, :i] = 1

            h_i = torch.cat((h, z.detach(), i), dim=1)
            z_i = torch.clamp(self.encoder2(h_i), min=-1.0, max=1.0)
            z[:, i:i+1] += z_i

        z = torch.view(batch_size, self.code_size, 1, 1)
        return z
