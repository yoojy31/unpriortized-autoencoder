import numpy as np
import torch
import torch.nn as nn
from ..__network__ import Network

class ARMDN(Network):
    forward_types = ('all', 'encoder', 'decoder')

    def __init__(self, args):
        super(ARMDN, self).__init__(args)
        self.armdn = None

    def build(self):
        assert (self.encoder and self.decoder) is None
        self.armdn = nn.Sequential()

    def forward(self, *x):
        z = torch.squeeze(torch.transpose(x[0], 1, 2), dim=3)
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
        def sample_gumbel(mu, sig, _pi):
            pi = nn.functional.gumbel_softmax(torch.squeeze(_pi), tau=tau)
            # rg = np.random.gumbel(loc=0, scale=1, size=pi.shape)
            # k = np.random.choice(self.args.n_gauss, size=(n_sample,), p=pi)

            mu = torch.squeeze(mu).detach().cpu().numpy()
            sig = torch.squeeze(sig).detach().cpu().numpy()
            pi = torch.squeeze(pi).detach().cpu().numpy()
            k = pi.argmax(axis=1)

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
            _z_i = sample_gumbel(mu, sig, _pi)

            one = torch.ones((n_sample, 1, 1)).cuda()
            _z_i_mask = torch.cat((_z_i, one), dim=1)
            seed = torch.cat((seed, _z_i_mask), dim=2)

        _z = seed[:, 0:1, self.args.z_size:(self.args.z_size * 2)]
        _z = torch.unsqueeze(torch.transpose(_z, 1, 2), dim=3)
        return _z
