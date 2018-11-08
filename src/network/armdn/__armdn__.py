import os
import math
import numpy as np
import torch
import torch.nn as nn
from ..__network__ import Network
from .__loss__ import mdn_loss_fn

class ARMDN(Network):
    def __init__(self, args):
        super(ARMDN, self).__init__(args)
        self.armdn = nn.Sequential()
        self.order = None
        self.order_name = self.__class__.__name__ + '.order'

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

    def calc_loss(self, z, mu, sig, pi):
        mdn_loss = mdn_loss_fn(z, mu, sig, pi)
        return mdn_loss

    def forward(self, z, tau=1.0):
        z = torch.squeeze(torch.transpose(z, 1, 2), dim=3)
        if self.args.ordering and (self.order is not None):
            z = self.realign_z(z)

        batch_size = z.size()[0]
        input_z_size = z.size()[2]
        input_p_size = 2 ** math.ceil(math.log2(self.args.z_size))

        pad = torch.zeros((batch_size, 1, input_p_size)).cuda()
        mask = torch.ones((batch_size, 1, input_p_size + input_z_size)).cuda()
        mask[:, :, :input_p_size] = 0

        pad.requires_grad_(True)
        mask.requires_grad_(True)

        z_pad = torch.cat((pad, z), dim=2)
        z_mask = torch.cat((z_pad, mask), dim=1)

        mog = self.armdn.forward(z_mask[:, :, :-1])
        mu, _sig, _pi = torch.split(mog, self.args.n_gauss, dim=1)
        sig, pi = torch.exp(_sig), self.softmax_with_tau(_pi, dim=1, tau=tau)

        mu = torch.unsqueeze(torch.transpose(mu, 1, 2), dim=3)
        sig = torch.unsqueeze(torch.transpose(sig, 1, 2), dim=3)
        pi = torch.unsqueeze(torch.transpose(pi, 1, 2), dim=3)
        return mu, sig, pi

    def sample(self, n_sample, tau=1.0):
        def sample_from_softmax(mu, sig, pi):
            mu = mu.detach().cpu().numpy()
            sig = sig.detach().cpu().numpy()
            pi = pi.detach().cpu().numpy()

            k = list()
            for pi_sample in pi:
                if self.args.n_gauss > 1:
                    k_sample = np.random.choice(self.args.n_gauss, p=pi_sample)
                else:
                    k_sample = 0
                k.append(k_sample)
            k = np.array(k)
            indices = (np.arange(n_sample), k)
            rn = np.random.randn(n_sample)

            z_i = rn * sig[indices] + mu[indices]
            z_i = torch.from_numpy(z_i).float().cuda()
            z_i = torch.reshape(z_i, (n_sample, 1, 1))
            return z_i

        z = torch.zeros((n_sample, self.args.z_size, 1, 1)).cuda()
        for i in range(self.args.z_size):
            input_z = z[:, :(i+1)]

            mu, sig, pi = self.forward(input_z, tau=tau)
            z_i = sample_from_softmax(mu[:, i, :, 0], sig[:, i, :, 0], pi[:, i, :, 0])
            z[:, i] = z_i

        if self.args.ordering and (self.order is not None):
            z = self.inv_realign_z(z)
        return z

    def softmax_with_tau(self, _pi, dim=1, tau=1.0):
        # softmax_with_temperature
        _pi = torch.exp(_pi / tau)
        pi = _pi / torch.sum(_pi, dim=dim, keepdim=True)
        return pi

    def save(self, save_dir):
        if super(ARMDN, self).save(save_dir):
            if self.order is not None:
                order_path = self.order_name = '.npy'
                np.save(order_path, self.order)
            return True
        else:
            return False

    def load(self, load_dir):
        if super(ARMDN, self).load(load_dir):
            order_path = self.order_name = '.npy'
            if os.path.exists(order_path):
                self.order = np.load(order_path)
            return True
        else:
            return False

