import os
from collections import OrderedDict

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from utils import timer
from ..__trainer__ import Trainer

class BasicSamplerTrainer0(Trainer):

    def __init__(self, encoder, decoder, sampler, optim, log_dir):
        super(BasicSamplerTrainer0, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.optim = optim

        self.num_bin = self.sampler.num_bin
        self.bin_size = 2.0 / self.num_bin
        self.code_size = self.sampler.code_size

        self.encoder_path = None
        self.decoder_path = None

        self.train_log_dir = os.path.join(log_dir, 'train')
        self.valid_log_dir = os.path.join(log_dir, 'valid')

    def forward(self, batch_dict, requires_grad):
        assert 's' in batch_dict.keys()

        s = batch_dict['s']
        m = torch.zeros(s.shape)

        s.requires_grad_(requires_grad)
        m.requires_grad_(requires_grad)

        s = s.cuda()
        m = m.cuda()

        for i in range(self.code_size):
            m[:, :i] = 1
            o_i = self.sampler.forward(torch.cat((s, m), dim=1).detach(), i)
            p_i = nn.functional.softmax(o_i)
            s_i = self.sampler.sample_z_1(p_i)
            s[:, i:i+1] = s_i
        # s = torch.reshape(s, (-1, self.code_size, 1, 1))
        return s

    def step(self, batch_dict, update):
        self.encoder.train(False)
        self.sampler.train(update)

        x = batch_dict['x']
        s = batch_dict['s']
        m = torch.zeros(s.size())

        x = x.requires_grad_(False).cuda()
        s = s.requires_grad_(update).cuda()
        m = m.requires_grad_(update).cuda()

        _, z = self.encoder.forward(x)
        z_lab = torch.squeeze(((z.detach() + 1) / self.bin_size).long())
        z_lab = torch.clamp(z_lab, min=0, max=self.num_bin-1)

        l_ce = 0
        for i in range(self.code_size):
            m[:, :i] = 1
            o_i = self.sampler.forward(torch.cat((s, m), dim=1).detach(), i)
            l_ce += nn.functional.cross_entropy(o_i, z_lab[:, i])
            s[:, i] = z[:, i]
        l_ce /= self.code_size

        if update:
            self.optim.zero_grad()
            l_ce.backward()
            self.optim.step()
            self.global_step += 1
        return z, l_ce

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

                train_time, (_, l_ce) = timer(self.step, batch_dict, True)
                l_ce = l_ce.item()

                value_dict = OrderedDict()
                value_dict['batching time'] = batch_time
                value_dict['training time'] = train_time
                value_dict['l_ce (train)'] = l_ce
                train_log_writer.add_scalar('l_ce', l_ce, self.global_step)

                if self.global_step % valid_intv == 0:
                    try:
                        batch_dict = next(valid_set_iter)
                    except StopIteration:
                        valid_set_iter = iter(valid_set_loader)
                        batch_dict = next(valid_set_iter)

                    z, l_ce = self.step(batch_dict, False)
                    s = self.forward(batch_dict, False)

                    # z_lab = torch.squeeze(((z.detach() + 1) / self.bin_size).long())
                    # s_lab = torch.squeeze(((s.detach() + 1) / self.bin_size).long())
                    # z_lab = torch.clamp(z_lab, min=0, max=self.num_bin-1)
                    # s_lab = torch.clamp(s_lab, min=0, max=self.num_bin-1)
                    z = z.cpu().detach().numpy()
                    s = s.cpu().detach().numpy()
                    l_ce = l_ce.item()

                    value_dict['l_ce (valid)'] = l_ce
                    valid_log_writer.add_scalar('l_ce', l_ce, self.global_step)
                    valid_log_writer.add_histogram('z_0.0', z[:, 0], self.global_step)
                    valid_log_writer.add_histogram('z_0.5', z[:, int(self.code_size/2)], self.global_step)
                    valid_log_writer.add_histogram('z_1.0', z[:, self.code_size-1], self.global_step)
                    valid_log_writer.add_histogram('s_0.0', s[:, 0], self.global_step)
                    valid_log_writer.add_histogram('s_0.5', s[:, int(self.code_size/2)], self.global_step)
                    valid_log_writer.add_histogram('s_1.0', s[:, self.code_size-1], self.global_step)
                yield value_dict

        train_log_writer.close()
        valid_log_writer.close()

    def use_cuda(self, cur_devise):
        torch.cuda.set_device(cur_devise)
        # when encoder and decoder is data paraller, what devise is current devise?
        self.encoder.cuda()
        self.decoder.cuda()
        self.sampler.cuda()
        self.is_cuda = True

    def decay_lr(self, decay_rate):
        for param_group in self.optim.param_groups:
            new_lr = param_group['lr'] * decay_rate
            param_group['lr'] = new_lr
        return new_lr

    def save_snapshot(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        pathes = BasicSamplerTrainer0.join_path(save_dir)

        if self.encoder_path is not None:
            os.symlink(self.encoder_path, pathes[0])
        else:
            torch.save(self.encoder.state_dict(), pathes[0])
        if self.decoder_path is not None:
            os.symlink(self.decoder_path, pathes[1])
        else:
            torch.save(self.decoder.state_dict(), pathes[1])

        torch.save(self.sampler.state_dict(), pathes[2])
        torch.save(self.optim.state_dict(), pathes[3])

    def load_snapshot(self, load_dir, learning_rate=None):
        pathes = BasicSamplerTrainer0.join_path(load_dir)
        self.encoder.load_state_dict(torch.load(pathes[0]))
        self.decoder.load_state_dict(torch.load(pathes[1]))

        if os.path.exists(pathes[2]):
            self.sampler.load_state_dict(torch.load(pathes[2]))
            self.optim.load_state_dict(torch.load(pathes[3]))

            # if learning_rate is not None:
            #     for param_group in self.optim.param_groups:
            #         param_group['lr'] = learning_rate

        self.encoder_path = pathes[0]
        self.decoder_path = pathes[1]

    @classmethod
    def join_path(cls, directory):
        encoder_path = os.path.join(directory, 'encoder.pth')
        decoder_path = os.path.join(directory, 'decoder.pth')
        sampler_path = os.path.join(directory, 'sampler.pth')
        optim_path = os.path.join(directory, 'optim.pth')
        return encoder_path, decoder_path, sampler_path, optim_path

class BasicSamplerTrainer1(BasicSamplerTrainer0):
    def forward(self, batch_dict, requires_grad):
        assert 's' in batch_dict.keys()

        batch_size = batch_dict['s'].size()[0]
        s = torch.zeros(
            (batch_size, 1, self.code_size * 2),
            requires_grad=requires_grad).cuda()
        m = torch.zeros(
            (batch_size, 1, self.code_size * 2),
            requires_grad=requires_grad).cuda()
        m[:, 0, self.code_size:] = 1

        for i in range(self.code_size):
            j = i + self.code_size

            m_i = m[:, 0:1, i:j]
            s_i = s[:, 0:1, i:j]
            f_i = torch.cat((s_i, m_i), dim=1).detach()

            p_i = self.sampler.forward(f_i, i)
            s_i = self.sampler.sample_z_1(p_i)
            s[:, 0, j] += s_i[:, 0, 0, 0]
        s = s[:, :, self.code_size:]
        s = torch.reshape(s, (batch_size, self.code_size, 1, 1))
        return s

    def step(self, batch_dict, update):
        self.encoder.train(False)
        self.sampler.train(update)

        x = batch_dict['x']
        x = x.requires_grad_(False).cuda()

        batch_size = x.size()[0]
        s = torch.zeros(
            (batch_size, 1, self.code_size * 2),
            requires_grad=update).cuda()
        m = torch.zeros(
            (batch_size, 1, self.code_size * 2),
            requires_grad=update).cuda()
        m[:, 0, self.code_size:] = 1

        _, z = self.encoder.forward(x)
        z_lab = torch.squeeze(((z.detach() + 1) / self.bin_size).long())
        z_lab = torch.clamp(z_lab, min=0, max=self.num_bin-1)

        l_ce = 0
        for i in range(self.code_size):
            j = i + self.code_size

            m_i = m[:, 0:1, i:j]
            s_i = s[:, 0:1, i:j]
            f_i = torch.cat((s_i, m_i), dim=1).detach()

            p_i = self.sampler.forward(f_i, i)
            l_ce += nn.functional.cross_entropy(p_i, z_lab[:, i])
            # print(s.size())
            # print(z.size())
            s[:, 0, j] = z[:, i, 0, 0]
        l_ce /= self.code_size

        if update:
            self.optim.zero_grad()
            l_ce.backward()
            self.optim.step()
            self.global_step += 1
        return z, l_ce
