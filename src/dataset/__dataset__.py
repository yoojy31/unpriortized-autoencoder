import abc
from torch.utils.data.dataset import Dataset

class ImgDataset(abc.ABC, Dataset):

    @abc.abstractmethod
    def pre_processing(self, x):
        pass

    @abc.abstractmethod
    def post_processing(self, x):
        pass
