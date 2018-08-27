import torch
import torch.nn as nn
from .basics import BasicEncoder00

class LatentHierarchicalEncoder00(BasicEncoder00):
    # mlp based hierarchical encoder

    def __init__(self, args):
        super(LatentHierarchicalEncoder00, self).__init__(args)

        self.mlp_list = list()
        for i in range(self.code_size):
            num_node = i + 1
            mlp = nn.Sequential(
                nn.Linear(num_node, num_node, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(num_node, num_node, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(num_node, 1, bias=False),
            )
            self.mlp_list.append(mlp)
        self.mlp_list = nn.Sequential(*self.mlp_list)

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

        for i in range(self.code_size):
            h_i = h[:, i:i+1]
            if i == 0:
                f_i = h_i
            else:
                z1_pre = z1[:, :i]
                f_i = torch.cat((h_i, z1_pre), dim=1)
            z1_i = self.mlp_list[i].forward(f_i)
            z1[:, i:i+1] = z1_i

        z2 = torch.clamp(z1, min=-1, max=1)
        z2 = torch.reshape(z2, (batch_size, self.code_size, 1, 1))
        return z1, z2
