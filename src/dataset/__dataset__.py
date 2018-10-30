import abc
from torch.utils.data.dataset import Dataset

class ImgDataset(abc.ABC, Dataset):
    @abc.abstractmethod
    def pre_process(self, x):
        pass

    @abc.abstractmethod
    def post_process(self, x):
        pass

    def augment(self, *x):
        pass
