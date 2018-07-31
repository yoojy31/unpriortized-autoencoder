import abc
from torch.utils.data.dataset import Dataset

class ImgDataset(abc.ABC, Dataset):

    @abc.abstractmethod
    def preprocessing(self, x):
        pass

    @abc.abstractmethod
    def inv_preprocessing(self, x):
        pass
