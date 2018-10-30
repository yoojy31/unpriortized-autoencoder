import os
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

    def forward(self, z):
        z = torch.squeeze(torch.transpose(z, 1, 2), dim=3)
        if self.args.ordering and (self.order is not None):
            z = self.realign_z(z)
        batch_size = z.size()[0]

        pad = torch.zeros((batch_size, 1, self.args.z_size)).cuda()
        mask = torch.ones((batch_size, 1, self.args.z_size * 2)).cuda()
        mask[:, :, :self.args.z_size] = 0

        pad.requires_grad_(True)
        mask.requires_grad_(True)

        z_pad = torch.cat((pad, z), dim=2)
        z_mask = torch.cat((z_pad, mask), dim=1)

        mog = self.armdn.forward(z_mask[:, :, :-1])
        mu, _sig, _pi = torch.split(mog, self.args.n_gauss, dim=1)
        sig, pi = torch.exp(_sig), nn.functional.softmax(_pi, dim=1)

        mu = torch.unsqueeze(torch.transpose(mu, 1, 2), dim=3)
        sig = torch.unsqueeze(torch.transpose(sig, 1, 2), dim=3)
        pi = torch.unsqueeze(torch.transpose(pi, 1, 2), dim=3)
        return mu, sig, pi

    def sample(self, n_sample, tau):
        def sample_from_softmax(mu, sig, _pi):
            # softmax_with_temperature
            _pi = torch.exp(_pi / tau)
            pi = _pi / torch.sum(_pi, dim=1, keepdim=True)

            mu = torch.squeeze(mu).detach().cpu().numpy()
            sig = torch.squeeze(sig).detach().cpu().numpy()
            pi = torch.squeeze(pi).detach().cpu().numpy()

            k = list()
            for pi_sample in pi:
                k_sample = np.random.choice(self.args.n_gauss, p=pi_sample)
                k.append(k_sample)
            k = np.array(k)
            indices = (np.arange(n_sample), k)
            rn = np.random.randn(n_sample)

            _z_i = rn * sig[indices] + mu[indices]
            _z_i = torch.from_numpy(_z_i).float().cuda()
            _z_i = torch.reshape(_z_i, (n_sample, 1, 1))
            return _z_i

        seed = torch.zeros((n_sample, 2, self.args.z_size)).cuda()
        for i in range(self.args.z_size):
            seed_i = seed[:, :, i:(i + self.args.z_size)]

            # print('[%d]' % i, seed_i.shape, type(seed_i), seed_i.is_cuda)
            mog = self.armdn.forward(seed_i)
            mu, _sig, _pi = torch.split(mog, self.args.n_gauss, dim=1)

            mu = torch.squeeze(mu)
            sig = torch.squeeze(torch.exp(_sig))
            _pi = torch.squeeze(_pi)
            _z_i = sample_from_softmax(mu, sig, _pi)

            one = torch.ones((n_sample, 1, 1)).cuda()
            _z_i_mask = torch.cat((_z_i, one), dim=1)
            seed = torch.cat((seed, _z_i_mask), dim=2)

        _z = seed[:, 0:1, self.args.z_size:(self.args.z_size * 2)]
        if self.args.ordering and (self.order is not None):
            _z = self.inv_realign_z(_z)
        _z = torch.unsqueeze(torch.transpose(_z, 1, 2), dim=3)
        return _z

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

