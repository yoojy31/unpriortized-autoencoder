import torch
import torch.nn as nn
from .__sampler__ import Sampler

class HierarchicalSampler00(Sampler):
    def __init__(self, args):
        super(HierarchicalSampler00, self).__init__(args)

        unit_len = max(self.code_size, self.num_bin)
        self.ar_sampler_list = list()
        for _ in range(args.code_size):
            ar_sampler = nn.Sequential(
                nn.Linear(args.code_size * 2, unit_len * 2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len * 2, unit_len, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len, args.num_bin, bias=False),
            )
            self.ar_sampler_list.append(ar_sampler)
        self.ar_sampler_list = nn.Sequential(*self.ar_sampler_list)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        idx = x[1]
        x = torch.squeeze(x[0])
        o = self.ar_sampler_list[idx].forward(x)
        p = nn.functional.softmax(o, dim=1)
        # o = torch.reshape(o, (-1, self.num_bin, 1, 1))
        return p

class HierarchicalSampler10(Sampler):
    def __init__(self, args):
        super(HierarchicalSampler10, self).__init__(args)

        unit_len = max(self.code_size, self.num_bin)
        self.ar_sampler_list = list()
        for _ in range(args.code_size):
            ar_sampler = nn.Sequential(
                nn.Linear(args.code_size * 2, unit_len * 2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len * 2, int(unit_len * 1.5), bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(int(unit_len * 1.5), int(unit_len * 1.5), bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(int(unit_len * 1.5), unit_len, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len, unit_len, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len, args.num_bin, bias=False),
            )
            self.ar_sampler_list.append(ar_sampler)
        self.ar_sampler_list = nn.Sequential(*self.ar_sampler_list)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        idx = x[1]
        x = torch.squeeze(x[0])
        o = self.ar_sampler_list[idx].forward(x)
        p = nn.functional.softmax(o, dim=1)
        # o = torch.reshape(o, (-1, self.num_bin, 1, 1))
        return p

class HierarchicalSampler11(Sampler):
    def __init__(self, args):
        super(HierarchicalSampler11, self).__init__(args)

        unit_len = max(self.code_size, self.num_bin)
        self.ar_sampler_list = list()
        for _ in range(args.code_size):
            ar_sampler = nn.Sequential(
                nn.Linear(args.code_size * 2, unit_len * 2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len * 2, int(unit_len * 1.5), bias=False),
                nn.BatchNorm1d(unit_len * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(int(unit_len * 1.5), int(unit_len * 1.5), bias=False),
                nn.BatchNorm1d(int(unit_len * 1.5), affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(int(unit_len * 1.5), unit_len, bias=False),
                nn.BatchNorm1d(unit_len, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len, unit_len, bias=False),
                nn.BatchNorm1d(unit_len, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(unit_len, args.num_bin, bias=False),
            )
            self.ar_sampler_list.append(ar_sampler)
        self.ar_sampler_list = nn.Sequential(*self.ar_sampler_list)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        self.train(mode=True)

        idx = x[1]
        x = torch.squeeze(x[0])
        o = self.ar_sampler_list[idx].forward(x)
        p = nn.functional.softmax(o, dim=1)
        # o = torch.reshape(o, (-1, self.num_bin, 1, 1))
        return p
