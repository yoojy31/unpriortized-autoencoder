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
            if not os.path.isdir(save_dir):
                raise IsADirectoryError(save_dir)
            else:
                pass
            file_name = self.__class__.__name__
            model_path = os.path.join(save_dir, file_name)
            torch.save(self.state_dict(), model_path)
        else:
            raise FileNotFoundError(save_dir)

    def load(self, load_dir):
        file_name = self.__class__.__name__
        model_path = os.path.join(load_dir, file_name)
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
        else:
            raise FileNotFoundError
