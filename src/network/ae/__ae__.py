from collections import OrderedDict
import torch
import torch.nn as nn
from ..__network__ import Network
from .__loss__ import PerceptualLoss

forward_types = ('all', 'encoder', 'decoder', 'autoencoder', 'loss')

class Autoencoder(Network):
    def __init__(self, args):
        super(Autoencoder, self).__init__(args)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        if self.args.perc_w != 0:
            self.perc_fn = PerceptualLoss()

    def calc_loss(self, _x, x):
        mse_loss = self.args.mse_w * nn.functional.mse_loss(_x, x)
        if self.args.perc_w == 0:
            perc_loss = torch.zeros(mse_loss.shape).cuda()
        else:
            perc_loss = self.args.perc_w * self.perc_fn.forward(_x, x)

        loss_dict = OrderedDict()
        loss_dict['mse'] = mse_loss
        loss_dict['perc'] = perc_loss
        return loss_dict

    def forward(self, x, _x=None, forward_type=forward_types[0]):
        assert (self.encoder and self.decoder) is not None
        assert forward_type in forward_types

        # all
        if forward_type == forward_types[0]:
            z = self.encoder.forward(x)
            _x = self.decoder.forward(z)
            result = self.calc_loss(_x, x)

        # encoder
        elif forward_type == forward_types[1]:
            result = self.encoder.forward(x)

        # decoder
        elif forward_type == forward_types[2]:
            z = x
            result = self.decoder.forward(z)

        # autoencoder
        elif forward_type == forward_types[3]:
            z = self.encoder.forward(x)
            result = self.decoder.forward(z)

        # loss
        elif forward_type == forward_types[4]:
            result = self.calc_loss(_x, x)

        else:
            result = None
        return result
