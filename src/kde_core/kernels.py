"""
Implementation of different kernel functions.

Yujia Li, 11/2014
"""

import numpy as np

def safe_diag(x):
    if isinstance(x, np.ndarray):
        return x.diagonal()
    if isinstance(x, np.array):
        if x.shape[0] > 4000:
            return np.array(x.asarray().diagonal())
        else:
            return x.diag()

    raise Exception()

class Kernel(object):
    def __init__(self):
        pass

    def compute_kernel_matrix(self, x):
        """
        x: n_examples * n_dims input data matrix

        Return: n_examples * n_examples kernel matrix
        """
        return self.compute_kernel_transformation(x, x)

    def compute_kernel_transformation(self, x_base, x_new):
        """
        x_base: n_examples_1 * n_dims data matrix
        x_new: n_examples_2 * n_dims data matrix

        For each example in x_new, compute its kernel distance with each of the
        examples in x_base, return a n_examples_2 * n_examples_1 matrix as the
        transformed representation of x_new.
        """
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def compute_kernel_matrix(self, x):
        x = x if isinstance(x, np.array) else np.array(x)
        xx = x.dot(x.T)
        x_diag = safe_diag(xx)

        return np.exp(-1.0 / (2 * self.sigma**2) * (-2 * xx + x_diag + x_diag[:, np.newaxis]))

    def compute_kernel_transformation(self, x_base, x_new):
        xx = x_new.dot(x_base.T)
        xx_base = (x_base**2).sum(axis=1)
        xx_new = (x_new**2).sum(axis=1)
        return np.exp(-1.0 / (2 * self.sigma**2) * (-2 * xx + xx_base + xx_new[:, np.newaxis]))

    def get_name(self):
        return 'gaussian_kernel'

class EuclideanKernel(Kernel):
    def __init__(self):
        pass

    def compute_kernel_matrix(self, x):
        x = x if isinstance(x, np.array) else np.array(x)
        xx = x.dot(x.T)
        x_diag = safe_diag(xx)

        return (-2 * xx + x_diag + x_diag[:, np.newaxis])

    def compute_kernel_transformation(self, x_base, x_new):
        xx = x_new.dot(x_base.T)
        xx_base = (x_base**2).sum(axis=1)
        xx_new = (x_new**2).sum(axis=1)

        return (-2 * xx + xx_base + xx_new[:, np.newaxis])

class CPUGaussianKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def compute_kernel_matrix(self, x):
        pass

class LinearKernel(Kernel):
    def compute_kernel_matrix(self, x):
        x = x if isinstance(x, np.array) else np.array(x)
        return x.dot(x.T)

    def compute_kernel_transformation(self, x_base, x_new):
        x_base = x_base if isinstance(x_base, np.array) else np.array(x_base)
        x_new = x_new if isinstance(x_new, np.array) else np.array(x_new)

        return x_new.dot(x_base.T)

    def get_name(self):
        return 'linear_kernel'

class CosineKernel(Kernel):
    def compute_kernel_matrix(self, x):
        x = x if isinstance(x, np.array) else np.array(x)
        x_norm = np.sqrt((x**2).sum(axis=1))
        x_norm = x_norm[:,np.newaxis] + x_norm[np.newaxis,:] + 1e-20

        return x.dot(x.T) / x_norm

    def compute_kernel_transformation(self, x_base, x_new):
        x_base = x_base if isinstance(x_base, np.array) else np.array(x_base)
        x_new = x_new if isinstance(x_new, np.array) else np.array(x_new)

        base_norm = (x_base**2).sum(axis=1)
        new_norm = (x_new**2).sum(axis=1)

        return x_new.dot(x_base.T) / (base_norm + new_norm[:,np.newaxis])

