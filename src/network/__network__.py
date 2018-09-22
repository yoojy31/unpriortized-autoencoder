import os
import abc
import torch
import torch.nn as nn

class Network(abc.ABC, nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args

    @ abc.abstractmethod
    def build(self):
        pass

    @ abc.abstractmethod
    def forward(self, *x):
        pass

    def save(self, save_dir):
        if os.path.exists(save_dir):
            file_name = self.__class__.__name__ + '.pth'
            model_path = os.path.join(save_dir, file_name)
            torch.save(self.state_dict(), model_path)
            return True
        else:
            return False

    def load(self, load_dir):
        file_name = self.__class__.__name__ + '.pth'
        model_path = os.path.join(load_dir, file_name)
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            return True
        else:
            return False
