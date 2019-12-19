import os
from collections import OrderedDict
import torch
import torch.nn as nn

import distribution
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

        self.encoder_name = self.__class__.__name__ + '.encoder'
        self.decoder_name = self.__class__.__name__ + '.decoder'

    def calc_loss(self, _x, x):
        pixel_loss = self.args.mse_w * nn.functional.mse_loss(_x, x)
        if self.args.perc_w == 0:
            perc_loss = torch.zeros(pixel_loss.shape).cuda()
        else:
            perc_loss = self.args.perc_w * self.perc_fn.forward(_x, x)

        loss_dict = OrderedDict()
        loss_dict['pixel'] = pixel_loss
        loss_dict['perc'] = perc_loss
        return loss_dict

    def posterior_pdf(self, z):
        return 1

    def forward(self, x, _x=None, forward_type=forward_types[0], dout=None):
        assert (self.encoder and self.decoder) is not None
        assert forward_type in forward_types

        # print(x.shape)

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

    def save(self, save_dir):
        if os.path.exists(save_dir):
            encoder_path = self.encoder_name + '.pth'
            decoder_path = self.decoder_name + '.pth'
            encoder_model_path = os.path.join(save_dir, encoder_path)
            decoder_model_path = os.path.join(save_dir, decoder_path)
            torch.save(self.encoder.state_dict(), encoder_model_path)
            torch.save(self.decoder.state_dict(), decoder_model_path)
            return True
        else:
            return False

    def load(self, load_dir):
        encoder_path = self.encoder_name + '.pth'
        decoder_path = self.decoder_name + '.pth'

        encoder_model_path = os.path.join(load_dir, encoder_path)
        decoder_model_path = os.path.join(load_dir, decoder_path)

        if os.path.exists(encoder_model_path) and os.path.exists(decoder_model_path):
            self.encoder.load_state_dict(torch.load(encoder_model_path))
            self.decoder.load_state_dict(torch.load(decoder_model_path))
            return True
        else:
            return False
