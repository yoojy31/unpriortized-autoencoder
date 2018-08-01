import torch
import torch.nn as nn
from .basics import BasicsEncoder00

class ARMLPEncoder00(BasicsEncoder00):
    # mlp based auto-regressive encoder

    def __init__(self, args):
        super(ARMLPEncoder00, self).__init__(args)

        self.ar_mlp = nn.Sequential(
            nn.Linear(args.code_size * 3, args.code_size * 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(args.code_size * 3, args.code_size * 2, bias=False),
            nn.BatchNorm1d(args.code_size, affine=True),
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
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                pass

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z = torch.zeros((batch_size, self.code_size))
        m = torch.zeros((batch_size, self.code_size))
        if h.is_cuda:
            z = z.cuda()
            m = m.cuda()

        for i in range(self.args.code_size):
            m[:, i] = 1

            h_i = torch.cat((h, z.detach(), m), dim=1)
            z_i = self.ar_mlp(h_i)
            z[:, i:i+1] += z_i

        z_i = nn.functional.tanh(z_i)
        z = torch.view(batch_size, self.code_size, 1, 1)
        return z

class ARMLPEncoder01(ARMLPEncoder00):
    def forward(self, *x):
        h = self.encoder1.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z = torch.zeros((batch_size, self.code_size))
        m = torch.zeros((batch_size, self.code_size))
        if h.is_cuda:
            z = z.cuda()
            m = m.cuda()

        for i in range(self.args.code_size):
            m[:, i] = 1

            h_i = torch.cat((h, z.detach(), m), dim=1)
            z_i = self.encoder2(h_i)
            z[:, i:i+1] += z_i

        z_i = nn.functional.tanh(z_i)
        z = torch.view(batch_size, self.code_size, 1, 1)
        return z
