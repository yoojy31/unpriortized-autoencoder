import numpy as np
import torch
import torch.nn as nn
from .basics import BasicEncoder00

class LatentAREncoder00(BasicEncoder00):
    # mlp based auto-regressive encoder
    # back-prop all

    def __init__(self, args):
        super(LatentAREncoder00, self).__init__(args)

        self.ar_mlp = nn.Sequential(
            nn.Linear(self.code_size * 3, self.code_size * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.code_size * 2, self.code_size * 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.code_size * 2, self.code_size, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.code_size, self.code_size, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.code_size, 1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z1 = torch.zeros((batch_size, self.code_size)).cuda()
        m = torch.zeros((batch_size, self.code_size)).cuda()

        for i in range(self.code_size):
            m[:, :i] = 1

            h_i = torch.cat((h, z1, m), dim=1)
            z1_i = self.ar_mlp(h_i)
            z1[:, i:i+1] += z1_i

        z1 = torch.clamp(z1, min=-1, max=1)
        z2 = torch.reshape(z1, (batch_size, self.code_size, 1, 1))
        return z1, z2

class LatentAREncoder01(LatentAREncoder00):
    # back-prop first

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z1 = torch.zeros((batch_size, self.code_size)).cuda()
        m = torch.zeros((batch_size, self.code_size)).cuda()

        for i in range(self.code_size):
            m[:, :i] = 1

            if i == 0:
                h_i = torch.cat((h, z1.detach(), m.detach()), dim=1)
            else:
                h_i = torch.cat((h.detach(), z1, m.detach()), dim=1)
            z1_i = self.ar_mlp(h_i)
            z1[:, i:i+1] += z1_i

        z2 = torch.clamp(z1, min=-1, max=1)
        z2 = torch.reshape(z2, (batch_size, self.code_size, 1, 1))
        return z1, z2

class LatentAREncoder02(LatentAREncoder00):
    # back-prop first, apply clamp each element

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z1 = torch.zeros((batch_size, self.code_size)).cuda()
        z2 = torch.zeros((batch_size, self.code_size)).cuda()
        m = torch.zeros((batch_size, self.code_size)).cuda()

        for i in range(self.code_size):
            m[:, :i] = 1

            if i == 0:
                h_i = torch.cat((h, z2.detach(), m.detach()), dim=1)
            else:
                h_i = torch.cat((h.detach(), z1, m.detach()), dim=1)
            z1_i = self.ar_mlp(h_i)
            z1[:, i:i+1] += z1_i
            z2[:, i:i+1] += torch.clamp(z1_i, min=-1, max=1)

        z2 = torch.reshape(z2, (batch_size, self.code_size, 1, 1))
        return z1, z2

class LatentAREncoder03(LatentAREncoder00):
    # back-prop all, tanh

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z = torch.zeros((batch_size, self.code_size)).cuda()
        m = torch.zeros((batch_size, self.code_size)).cuda()

        for i in range(self.code_size):
            m[:, :i] = 1

            h_i = torch.cat((h, z, m), dim=1)
            z_i = self.ar_mlp(h_i)
            z[:, i:i+1] += z_i

        z = nn.functional.tanh(z)
        z = torch.reshape(z, (batch_size, self.code_size, 1, 1))
        return z

class LatentAREncoder10(LatentAREncoder00):
    # back-prop all, clamp, sliding window

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        s = torch.zeros((batch_size, self.code_size * 2)).cuda()
        m = torch.zeros((batch_size, self.code_size * 2)).cuda()
        m[:, self.code_size:] = 1

        for i in range(self.code_size):
            j = i + self.code_size

            m_i = m[:, i:j]
            s_i = s[:, i:j]
            f_i = torch.cat((h, s_i, m_i), dim=1)

            z_i = self.ar_mlp(f_i)
            s[:, j:j+1] += z_i

        z = s[:, self.code_size:self.code_size * 2]
        z = torch.clamp(z, min=-1, max=1)
        z = torch.reshape(z, (batch_size, self.code_size, 1, 1))
        return z

class LatentAREncoder11(LatentAREncoder00):
    # back-prop first, clamp, sliding window

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        s = torch.zeros((batch_size, self.code_size * 2)).cuda()
        m = torch.zeros((batch_size, self.code_size * 2)).cuda()
        m[:, self.code_size:] = 1

        for i in range(self.code_size):
            j = i + self.code_size

            m_i = m[:, i:j]
            s_i = s[:, i:j]
            if i == 0:
                f_i = torch.cat((h, s_i, m_i), dim=1)
            else:
                f_i = torch.cat((h.detach(), s_i, m_i), dim=1)

            z_i = self.ar_mlp(f_i)
            s[:, j:j+1] += z_i

        z = s[:, self.code_size:self.code_size * 2]
        z = torch.clamp(z, min=-1, max=1)
        z = torch.reshape(z, (batch_size, self.code_size, 1, 1))
        return z

class LatentAREncoder20(BasicEncoder00):
    # backward-all, satlins, sliding window, using conv

    def __init__(self, args):
        super(LatentAREncoder20, self).__init__(args)
        num_blocks = int(np.log2(self.code_size))

        self.ar_mlp = list()
        for i in range(num_blocks):
            if i == 0:
                self.ar_mlp.append(nn.Conv1d(3, 64, 4, 2, padding=1, bias=True))
                self.ar_mlp.append(nn.LeakyReLU(0.2))
            elif i == (num_blocks - 1):
                self.ar_mlp.append(nn.Conv1d(64, 1, 4, 2, padding=1,  bias=True))
            else:
                self.ar_mlp.append(nn.Conv1d(64, 64, 4, 2, padding=1, bias=True))
                # self.ar_mlp.append(nn.BatchNorm1d(64, affine=True))
                self.ar_mlp.append(nn.LeakyReLU(0.2))
        self.ar_mlp = nn.Sequential(*self.ar_mlp)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        # self.train(mode=True)
        batch_size = x[0].size()[0]

        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)
        h = torch.reshape(h, (batch_size, 1, self.code_size))

        s = torch.zeros((batch_size, 1, self.code_size * 2)).cuda()
        m = torch.zeros((batch_size, 1, self.code_size * 2)).cuda()
        m[:, 0, self.code_size:] = 1

        for i in range(self.code_size):
            j = i + self.code_size

            m_i = m[:, 0:1, i:j]
            s_i = s[:, 0:1, i:j]
            f_i = torch.cat((h, s_i, m_i), dim=1)
            # if i == 0:    f_i = torch.cat((h, s_i, m_i), dim=1)
            # else:         f_i = torch.cat((h.detach(), s_i, m_i), dim=1)

            z_i = self.ar_mlp(f_i)
            s[:, 0:1, j:j+1] += z_i

        z1 = s[:, 0, self.code_size:self.code_size * 2]
        z2 = torch.clamp(z1, min=-1, max=1)
        z2 = torch.reshape(z2, (batch_size, self.code_size, 1, 1))
        return z1, z2

class LatentAREncoder21(LatentAREncoder20):
    # backward-first, satlins, sliding window, using conv

    def forward(self, *x):
        # self.train(mode=True)
        batch_size = x[0].size()[0]

        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)
        h = torch.reshape(h, (batch_size, 1, self.code_size))

        s = torch.zeros((batch_size, 1, self.code_size * 2)).cuda()
        m = torch.zeros((batch_size, 1, self.code_size * 2)).cuda()
        m[:, 0, self.code_size:] = 1

        for i in range(self.code_size):
            j = i + self.code_size

            m_i = m[:, 0:1, i:j]
            s_i = s[:, 0:1, i:j]
            if i == 0:
                f_i = torch.cat((h, s_i, m_i), dim=1)
            else:
                f_i = torch.cat((h.detach(), s_i, m_i), dim=1)

            z_i = self.ar_mlp(f_i)
            s[:, 0:1, j:j+1] += z_i

        z1 = s[:, 0, self.code_size:self.code_size * 2]
        z2 = torch.clamp(z1, min=-1, max=1)
        z2 = torch.reshape(z2, (batch_size, self.code_size, 1, 1))
        return z1, z2

class LatentAREncoder30(LatentAREncoder20):
    # backward-all, satlins, fixed pos, using conv, pass element (scalar) of encoder's output

    def __init__(self, args):
        super(LatentAREncoder30, self).__init__(args)
        num_blocks = int(np.log2(self.code_size))

        self.ar_conv = list()
        for i in range(num_blocks):
            if i == 0:
                self.ar_conv.append(nn.Conv1d(2, 64, 4, 2, padding=1, bias=True))
                self.ar_conv.append(nn.LeakyReLU(0.2))
            elif i == (num_blocks - 1):
                self.ar_conv.append(nn.Conv1d(64, 1, 4, 2, padding=1,  bias=True))
            else:
                self.ar_conv.append(nn.Conv1d(64, 64, 4, 2, padding=1, bias=True))
                # self.ar_mlp.append(nn.BatchNorm1d(64, affine=True))
                self.ar_conv.append(nn.LeakyReLU(0.2))
        self.ar_conv = nn.Sequential(*self.ar_conv)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *x):
        # self.train(mode=True)
        batch_size = x[0].size()[0]

        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)
        h = torch.reshape(h, (batch_size, 1, self.code_size))

        z1 = torch.zeros((batch_size, 1, self.code_size)).cuda()
        m = torch.zeros((batch_size, 1, self.code_size + 1)).cuda()

        for i in range(self.code_size):
            m[:, 0:1, :i+1] = 1
            h_i = h[:, 0:1, i:i+1]

            f_i = torch.cat((torch.cat((h_i, z1), dim=2), m), dim=1)
            z1_i = self.ar_conv(f_i)
            z1[:, 0:1, i:i+1] += z1_i

        z2 = torch.clamp(z1, min=-1, max=1)
        z2 = torch.reshape(z2, (batch_size, self.code_size, 1, 1))
        return z1, z2
