import torch.nn as nn
from .__ae__ import forward_types
from .ae01 import Autoencoder01

class Autoencoder10(Autoencoder01):
    # autoencoder with dropout

    def dropout(self, z, dout):
        z = nn.functional.dropout2d(z, p=dout, training=True)
        return z

    def forward(self, x, _x=None, forward_type=forward_types[0], dout=None):
        assert (self.encoder and self.decoder) is not None
        assert forward_type in forward_types
        if dout is None:
            dout = self.args.z_dout_rate

        # all
        if forward_type == forward_types[0]:
            z = self.encoder.forward(x)
            z = nn.functional.dropout2d(z, p=dout, training=True)
            _x = self.decoder.forward(z)
            result = self.calc_loss(_x, x)

        # encoder
        elif forward_type == forward_types[1]:
            z = self.encoder.forward(x)
            result = nn.functional.dropout2d(z, p=dout, training=True)

        # decoder
        elif forward_type == forward_types[2]:
            z = x
            result = self.decoder.forward(z)

        # autoencoder
        elif forward_type == forward_types[3]:
            z = self.encoder.forward(x)
            z = nn.functional.dropout(z, p=dout, training=True)
            result = self.decoder.forward(z)

        # loss
        elif forward_type == forward_types[4]:
            result = self.calc_loss(_x, x)

        else:
            result = None
        return result
