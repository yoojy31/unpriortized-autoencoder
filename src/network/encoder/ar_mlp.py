import torch
import torch.nn as nn
from .basics import BasicEncoder00

class ARMLPEncoder00(BasicEncoder00):
    # mlp based auto-regressive encoder
    # back-prop all

    def __init__(self, args):
        super(ARMLPEncoder00, self).__init__(args)

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
        z = torch.zeros((batch_size, self.code_size)).cuda()
        m = torch.zeros((batch_size, self.code_size)).cuda()

        for i in range(self.code_size):
            m[:, :i] = 1

            h_i = torch.cat((h, z, m), dim=1)
            z_i = self.ar_mlp(h_i)
            z[:, i:i+1] += z_i

        z = torch.clamp(z, min=-1, max=1)
        z = torch.reshape(z, (batch_size, self.code_size, 1, 1))
        return z

class ARMLPEncoder01(ARMLPEncoder00):
    # back-prop first

    def forward(self, *x):
        h = self.encoder.forward(x[0])
        h = torch.squeeze(h)

        batch_size = h.size()[0]
        z = torch.zeros((batch_size, self.code_size)).cuda()
        m = torch.zeros((batch_size, self.code_size)).cuda()

        for i in range(self.code_size):
            m[:, :i] = 1

            if i == 0:
                h_i = torch.cat((h, z, m), dim=1)
            else:
                h_i = torch.cat((h.detach(), z, m), dim=1)
            z_i = self.ar_mlp(h_i)
            z[:, i:i+1] += z_i

        z = torch.clamp(z, min=-1, max=1)
        z = torch.reshape(z, (batch_size, self.code_size, 1, 1))
        return z

class ARMLPEncoder10(ARMLPEncoder00):
    # back-prop all

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

class ARMLPEncoder11(ARMLPEncoder00):
    # back-prop first

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
