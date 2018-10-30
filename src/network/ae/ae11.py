import torch
import torch.nn as nn
import distribution
from .__ae__ import forward_types
from .ae00 import Autoencoder00

class Autoencoder11(Autoencoder00):
    # autoencoder with partial dropout

    def dropout(self, z, dout):
        z1 = z[:, :self.args.static_z_size, :, :]
        z2 = z[:, self.args.static_z_size:, :, :]
        z2 = nn.functional.dropout2d(z2, p=dout, training=True)
        z = torch.cat((z1, z2), dim=1)
        return z

    def posterior_pdf(self, z):
        z2 = z[:, self.args.static_z_size:, :, :]
        mask = torch.where(z2 == 0, 0, 1)
        return torch.exp(distribution.log_bernoulli_pdf(0.5, mask))

    def forward(self, x, _x=None, forward_type=forward_types[0], dout=None):
        assert (self.encoder and self.decoder) is not None
        assert forward_type in forward_types
        dout_rate = dout if dout is not None else self.args.z_dout_rate

        # all
        if forward_type == forward_types[0]:
            z = self.encoder.forward(x)
            z = self.dropout(z, dout_rate)
            _x = self.decoder.forward(z)
            result = self.calc_loss(_x, x)

        # encoder
        elif forward_type == forward_types[1]:
            z = self.encoder.forward(x)
            result = self.dropout(z, dout_rate)

        # decoder
        elif forward_type == forward_types[2]:
            z = x
            result = self.decoder.forward(z)

        # autoencoder
        elif forward_type == forward_types[3]:
            z = self.encoder.forward(x)
            z = self.dropout(z, dout_rate)
            result = self.decoder.forward(z)

        # loss
        elif forward_type == forward_types[4]:
            result = self.calc_loss(_x, x)

        else:
            result = None
        return result
