import numpy as np
import torch
import torch.nn as nn
from .__sampler__ import Sampler

class ARSampler00(Sampler):
    def __init__(self, args):
        super(ARSampler00, self).__init__(args)

        unit_len = max(self.code_size, self.num_bin)
        self.ar_sampler = nn.Sequential(
            nn.Linear(self.code_size * 2, unit_len * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, unit_len, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, self.num_bin, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        x = torch.squeeze(x[0])
        # print(x.shape)
        o = self.ar_sampler.forward(x)
        p = nn.functional.softmax(o, dim=1)
        # o = torch.reshape(o, (-1, self.num_bin, 1, 1))
        return p

class ARSampler01(ARSampler00):
    def __init__(self, args):
        super(ARSampler01, self).__init__(args)

        unit_len = max(self.code_size, self.num_bin)
        self.ar_sampler = nn.Sequential(
            nn.Linear(self.code_size * 2, unit_len * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len * 2, bias=False),
            nn.BatchNorm1d(unit_len * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len, bias=False),
            nn.BatchNorm1d(unit_len, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, unit_len, bias=False),
            nn.BatchNorm1d(unit_len, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, self.num_bin, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

class ARSampler02(ARSampler00):
    def __init__(self, args):
        super(ARSampler02, self).__init__(args)

        unit_len = max(self.code_size, self.num_bin)
        self.ar_sampler = nn.Sequential(
            nn.Linear(self.code_size * 2, unit_len * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len * 2, bias=False),
            nn.BatchNorm1d(unit_len * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len, bias=False),
            nn.BatchNorm1d(unit_len, affine=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, unit_len, bias=False),
            nn.BatchNorm1d(unit_len, affine=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, self.num_bin, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

class ARSampler10(ARSampler00):
    def __init__(self, args):
        super(ARSampler10, self).__init__(args)

        unit_len = max(self.code_size, self.num_bin)
        self.ar_sampler = nn.Sequential(
            nn.Linear(self.code_size * 2, unit_len * 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 3, unit_len * 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 3, unit_len * 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 3, unit_len * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len * 2, unit_len, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, unit_len, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, unit_len, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(unit_len, self.num_bin, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

class ARSampler20(Sampler):
    # conv architecture, sliding window

    def __init__(self, args):
        super(ARSampler20, self).__init__(args)
        num_blocks = int(np.log2(self.code_size))

        self.ar_conv = list()
        for i in range(num_blocks):
            if i == 0:
                self.ar_conv.append(nn.Conv1d(2, 64, 4, 2, padding=1, bias=True))
                self.ar_conv.append(nn.LeakyReLU(0.2))
            elif i == (num_blocks - 1):
                self.ar_conv.append(nn.Conv1d(64, self.num_bin, 4, 2, padding=1,  bias=True))
            else:
                self.ar_conv.append(nn.Conv1d(64, 64, 4, 2, padding=1, bias=True))
                self.ar_conv.append(nn.BatchNorm1d(64, affine=True))
                self.ar_conv.append(nn.LeakyReLU(0.2))
        self.ar_conv = nn.Sequential(*self.ar_conv)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        self.train(mode=True)
        o = self.ar_conv.forward(x[0])
        o = torch.squeeze(o)
        # p = nn.functional.softmax(o, dim=1)
        return o
