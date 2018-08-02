import os
from collections import OrderedDict

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from utils import timer
from ..__trainer__ import Trainer

class BagicsAETrainer0(Trainer):
    # autoencoder trainer

    def __init__(self, encoder, decoder, optim, log_dir):
        super(BagicsAETrainer0, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optim = optim

        self.train_log_dir = os.path.join(log_dir, 'train')
        self.valid_log_dir = os.path.join(log_dir, 'valid')

    def forward(self, batch_dict, requires_grad):
        assert 'x' in batch_dict.keys()

        x = batch_dict['x'].cuda()
        x.requires_grad_(requires_grad)

        self.encoder.train(True)
        self.decoder.train(True)

        z = self.encoder(x)
        _x = self.decoder(z)
        return x, z, _x

    def step(self, batch_dict, update):
        x, z, _x = self.forward(batch_dict, update)
        l_mse = nn.functional.mse_loss(_x, x)

        if update:
            self.optim.zero_grad()
            l_mse.backward()
            self.optim.step()
            self.global_step += 1
        return x, z, _x, l_mse

    def train(self, train_set_loader, valid_set_loader,
              start_epoch, finish_epoch, valid_intv=10):

        if not os.path.exists(self.train_log_dir):
            os.mkdir(self.train_log_dir)
        if not os.path.exists(self.valid_log_dir):
            os.mkdir(self.valid_log_dir)

        train_log_writer = SummaryWriter(self.train_log_dir)
        valid_log_writer = SummaryWriter(self.valid_log_dir)

        num_batch = train_set_loader.__len__()
        self.global_step = start_epoch * num_batch
        for _ in range(start_epoch, finish_epoch):

            train_set_iter = iter(train_set_loader)
            valid_set_iter = iter(valid_set_loader)
            for __ in range(num_batch):
                try:
                    batch_time, batch_dict = \
                        timer(next, train_set_iter)
                except StopIteration:
                    print('Training set: stop iteration\n')

                train_time, (_, _, _, l_mse) = timer(self.step, batch_dict, True)

                l_mse = l_mse.item()
                train_log_writer.add_scalar('l_mse', l_mse, self.global_step)

                value_dict = OrderedDict()
                value_dict['batching time'] = batch_time
                value_dict['training time'] = train_time
                value_dict['l_mse (train)'] = l_mse

                if self.global_step % valid_intv == 0:
                    try:
                        batch_dict = next(valid_set_iter)
                    except StopIteration:
                        valid_set_iter = iter(valid_set_loader)
                        batch_dict = next(valid_set_iter)

                    _, z, _, l_mse = self.step(batch_dict, False)
                    z = z.cpu().detach().numpy()
                    l_mse = l_mse.item()

                    valid_log_writer.add_scalar('l_mse', l_mse, self.global_step)
                    valid_log_writer.add_histogram('z', z, self.global_step)
                    valid_log_writer.add_histogram('z_0', z[:, 0], self.global_step)
                    valid_log_writer.add_histogram('z_n', z[:, -1], self.global_step)
                    value_dict['l_mse (valid)'] = l_mse

                yield value_dict

        train_log_writer.close()
        valid_log_writer.close()

    def use_cuda(self, cur_devise):
        torch.cuda.set_device(cur_devise)
        # when encoder and decoder is data paraller, what devise is current devise?
        self.encoder.cuda()
        self.decoder.cuda()
        self.is_cuda = True

    def decay_lr(self, decay_rate):
        for param_group in self.optim.param_groups:
            new_lr = param_group['lr'] * decay_rate
            param_group['lr'] = new_lr
        return new_lr

    def save_snapshot(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        pathes = BagicsAETrainer0.join_path(save_dir)
        torch.save(self.encoder.state_dict(), pathes[0])
        torch.save(self.decoder.state_dict(), pathes[1])
        torch.save(self.optim.state_dict(), pathes[2])

    def load_snapshot(self, load_dir, learning_rate=None):
        pathes = BagicsAETrainer0.join_path(load_dir)
        self.encoder.load_state_dict(torch.load(pathes[0]))
        self.decoder.load_state_dict(torch.load(pathes[1]))
        self.optim.load_state_dict(torch.load(pathes[2]))

        if learning_rate is not None:
            for param_group in self.optim.param_groups:
                param_group['lr'] = learning_rate

    @classmethod
    def join_path(cls, _dir):
        encoder_path = os.path.join(_dir, 'encoder.pth')
        decoder_path = os.path.join(_dir, 'decoder.pth')
        optim_path = os.path.join(_dir, 'optimizer.pth')
        return encoder_path, decoder_path, optim_path
