import torch.nn as nn
from ..__network__ import Network

class Autoencoder(Network):
    forward_types = ('all', 'encoder', 'decoder')

    def __init__(self, args):
        super(Autoencoder, self).__init__(args)
        self.encoder = None
        self.decoder = None

    def build(self):
        assert (self.encoder and self.decoder) is None
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

    # def forward(self, *x):
    #     return self.forward(x[0], forward_type=x[1])

    def forward(self, x, forward_type=forward_types[0]):
        assert (self.encoder and self.decoder) is not None
        assert forward_type in Autoencoder.forward_types

        # all
        if forward_type == Autoencoder.forward_types[0]:
            z = self.encoder(x)
            return self.decoder(z)
        elif forward_type == Autoencoder.forward_types[1]:
            return self.encoder(x)
        elif forward_type == Autoencoder.forward_types[2]:
            z = x
            return self.decoder(z)
        else:
            return None
