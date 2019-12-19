import os
import math
import numpy as np
import torch
import torch.nn as nn
from ..__network__ import Network

class GMMN(Network):
    def __init__(self, args):
        super(GMMN, self).__init__(args)
        self.gmmn = nn.Sequential()
        self.order = None
        self.order_name = self.__class__.__name__ + '.order'

    def build(self):
        self.gmmn = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.ReLU(),
            nn.Linear(784, self.args.z_size))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.00, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                pass

    def set_order(self, order):
        self.order = order

    def realign_z(self, z):
        if self.order is not None:
            z_realign = z.clone()
            for i, o_i in enumerate(self.order):
                z_realign[o_i] = z[i]
            return z_realign

    def inv_realign_z(self, z_realign):
        if self.order is not None:
            z = z_realign.clone()
            for i, o_i in enumerate(self.order):
                z[i] = z_realign[o_i]
            return z

    def calc_loss(self, _z, z, sigma=[0.001]):
        def get_scale_matrix(M, N):
            s1 = (torch.ones((N, 1)) * 1.0 / N).cuda()
            s2 = (torch.ones((M, 1)) * -1.0 / M).cuda()
            return torch.cat((s1, s2), 0)

        z = torch.squeeze(z)
        _z = torch.squeeze(_z)

        Z = torch.cat((_z, z), 0)
        ZZ = torch.matmul(Z, Z.t())
        Z2 = torch.sum(Z * Z, 1, keepdim=True)
        exp = ZZ - 0.5 * Z2 - 0.5 * Z2.t()

        M = _z.size()[0]
        N = z.size()[0]
        s = get_scale_matrix(M, N)
        S = torch.matmul(s, s.t())

        loss = 0
        for v in sigma:
            kernel_val = torch.exp(exp / v)
            loss += torch.sum(S * kernel_val)

        loss = torch.sqrt(loss)
        return loss

    def forward(self, n_samples, tau=1.0):
        shape = (n_samples, 10)
        u = np.random.uniform(low=-1.0, high=1.0, size=shape)
        u = torch.from_numpy(u).float().cuda() / tau
        _z = self.gmmn.forward(u)
        _z = torch.reshape(_z, shape=(n_samples, self.args.z_size, 1, 1))
        return _z

    def sample(self, n_samples, tau=1.0):
        _z = self.forward(n_samples, tau)
        return _z

    def save(self, save_dir):
        if super(GMMN, self).save(save_dir):
            if self.order is not None:
                order_path = self.order_name = '.npy'
                np.save(order_path, self.order)
            return True
        else:
            return False

    def load(self, load_dir):
        if super(GMMN, self).load(load_dir):
            order_path = self.order_name = '.npy'
            if os.path.exists(order_path):
                self.order = np.load(order_path)
            return True
        else:
            return False

