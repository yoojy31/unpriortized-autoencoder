import abc

class Trainer(abc.ABC):
    def __init__(self):
        self.global_step = 0
        self.is_cuda = False

    @abc.abstractmethod
    def forward(self, batch_dict, requires_grad):
        pass

    @abc.abstractmethod
    def step(self, batch_dict, update):
        pass

    @abc.abstractmethod
    def train(self, train_set_loader, valid_set_loader,
              start_epoch, finish_epoch, valid_intv=10):
        pass

    @abc.abstractmethod
    def use_cuda(self, cur_devise):
        pass

    @abc.abstractmethod
    def save_snapshot(self, save_dir):
        pass

    @abc.abstractmethod
    def load_snapshot(self, load_dir, learning_rate=None):
        pass
