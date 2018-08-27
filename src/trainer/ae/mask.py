import numpy as np
import torch
import torch.nn as nn
from .basics import BasicAETrainer0

class MaskAETrainer0(BasicAETrainer0):
    # masked autoencoder trainer

    def forward(self, batch_dict, requires_grad):
        assert 'x' in batch_dict.keys()

        x = batch_dict['x']
        x = x.requires_grad_(requires_grad).cuda()

        z1, z2 = self.encoder(x)

        if requires_grad:
            batch_size = x.size()[0]
            code_size = self.encoder.code_size
            rand_u = np.random.randint(code_size, size=code_size)

            m = torch.zeros(batch_size, code_size, 1, 1)
            m = m.requires_grad_(requires_grad).cuda()
            for i in range(code_size):
                m[i, :int(rand_u[i]+1)] = 1

            z2_m = z2 * m
            _x = self.decoder(z2_m)
        else:
            _x = self.decoder(z2)

        return x, z1, z2, _x

    def step(self, batch_dict, update):
        self.encoder.train(update)
        self.decoder.train(update)

        x, z1, z2, _x = self.forward(batch_dict, update)
        l_mse = nn.functional.mse_loss(_x, x)

        if update:
            self.optim.zero_grad()
            l_mse.backward()
            self.optim.step()
            self.global_step += 1
        return x, z1, z2, _x, l_mse
