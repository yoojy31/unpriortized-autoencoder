"""
Module for evaluating MMD generative models.
Yujia Li, 11/2014
"""

import math
import _pickle as pickle
import time
import numpy as np
import scipy.misc
from tqdm import tqdm
# import gnumpy as gnp
# import kde_core.generative as gen
import kde_core.kernels as ker
import parzen
import torch

def log_exp_sum_1d(x):
    """
    This computes log(exp(x_1) + exp(x_2) + ... + exp(x_n)) as
    x* + log(exp(x_1-x*) + exp(x_2-x*) + ... + exp(x_n-x*)), where x* is the
    max over all x_i.  This can avoid numerical problems.
    """
    x_max = x.max()
    return x_max + np.log(np.exp(x - x_max).sum())

def log_exp_sum(x, axis=1):
    x_max = x.max(axis=axis)
    return (x_max + np.log(np.exp(x - x_max[:,np.newaxis]).sum(axis=axis)))# .asarray()

class KDE(object):
    """
    Kernel density estimation.
    """
    def __init__(self, data, sigma):
        self.x = data
        self.sigma = sigma
        self.N = self.x.shape[0]
        self.d = self.x.shape[1]
        self._ek =  ker.EuclideanKernel()
        # self._ek = ker.GaussianKernel(self.sigma)

        self.factor = float(-np.log(self.N) - self.d / 2.0 * np.log(2 * np.pi * self.sigma**2))

    def _log_likelihood(self, data):
        return log_exp_sum(-self._ek.compute_kernel_transformation(self.x, data) / (2 * self.sigma**2), axis=1) + self.factor

    def log_likelihood(self, data, batch_size=1000):
        n_cases = data.shape[0]
        if n_cases <= batch_size:
            return self._log_likelihood(data)
        else:
            n_batches = int(math.ceil(n_cases / batch_size))
            # n_batches = int((n_cases + batch_size - 1) / batch_size)
            log_like = np.zeros(n_cases, dtype=np.float)

            for i_batch in range(n_batches):
                i_start = i_batch * batch_size
                i_end = n_cases if (i_batch + 1 == n_batches) else (i_start + batch_size)
                batch_log_like = self._log_likelihood(data[i_start:i_end])
                log_like[i_start:i_end] = batch_log_like
            return log_like

    def likelihood(self, data):
        """
        data is a n_example x n_dims matrix.
        """
        return np.exp(self.log_likelihood(data))

    def average_likelihood(self, data):
        return self.likelihood(data).mean()

    def average_log_likelihood(self, data, batch_size=1000):
        return self.log_likelihood(data, batch_size=batch_size).mean()

    def average_std_log_likelihood(self, data, batch_size=1000):
        l = self.log_likelihood(data)
        return l.mean(), l.std()

    def average_se_log_likelihood(self, test_data, batch_size=1000):
        l = self.log_likelihood(test_data, batch_size=batch_size)
        return l.mean(), l.std() / np.sqrt(test_data.shape[0])

class AlternativeKDE(object):
    """
    Kernel density estimation.
    """
    def __init__(self, x, sigma):
        self.sigma = sigma
        self.x = x
        self.N = self.x.shape[0]
        self.d = self.x.shape[1]

    def _compute_log_prob(self, data, batch_size=1000):
        """
        Break down data into smaller pieces so large matrix will also work.
        """
        n_cases = 500#data.shape[0]
        K = np.zeros((n_cases, self.N), dtype=np.float)
        log_prob = np.zeros(n_cases, dtype=np.float)
        for i in range(n_cases):
            K[i] = -((self.x - data[i])**2).sum(axis=1) / (2 * self.sigma**2)
            log_prob[i] = log_exp_sum_1d(K[i]) - np.log(self.N) - self.d / 2.0 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))
        return log_prob

    def likelihood(self, data):
        """
        data is a n_example x n_dims matrix.
        """
        return np.exp(self._compute_log_prob(data))

    def average_likelihood(self, data):
        return self.likelihood(data).mean()

    def log_likelihood(self, data):
        return self._compute_log_prob(data)

    def average_log_likelihood(self, data):
        return self.log_likelihood(data).mean()

    def average_se_log_likelihood(self, data, batch_size=1000):
        l = self._compute_log_prob(data, batch_size=batch_size)
        return l.mean(), l.std() / np.sqrt(data.shape[0])

class TorchAlternativeKDE(object):
    """
    Kernel density estimation.
    """
    def __init__(self, data, sigma):
        self.x = data
        self.sigma = sigma
        self.N = self.x.shape[0]
        self.d = self.x.shape[1]

    def _compute_log_prob(self, data, batch_size=1000):
        def torch_log_exp_sum(x):
            """
            This computes log(exp(x_1) + exp(x_2) + ... + exp(x_n)) as
            x* + log(exp(x_1-x*) + exp(x_2-x*) + ... + exp(x_n-x*)), where x* is the
            max over all x_i.  This can avoid numerical problems.
            """
            x_max = x.max()
            return x_max + torch.log(torch.exp(x - x_max).sum())

        """
        Break down data into smaller pieces so large matrix will also work.
        """
        n_cases = data.shape[0]
        K = torch.zeros((n_cases, self.N)).float().cuda()
        log_prob = torch.zeros(n_cases).float().cuda()
        for i in range(n_cases):
            K[i] = -((self.x - data[i])**2).sum(dim=1) / (2 * self.sigma**2)
            log_prob[i] = torch_log_exp_sum(K[i]) - math.log(self.N) - self.d / 2.0 * (math.log(2 * math.pi) + 2 * math.log(self.sigma))
        return log_prob

    def average_se_log_likelihood(self, data, batch_size=1000):
        l = self._compute_log_prob(data, batch_size=batch_size)
        return l.mean(), l.std() / np.sqrt(data.shape[0])


def kde_evaluation(test_data, samples, sigma_range=np.arange(0.1, 0.3, 0.5), verbose=True):
    best_log_likelihood = float('-inf')
    for sigma in sigma_range:
        log_likelihood = KDE(samples, sigma).average_log_likelihood(test_data)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
        if verbose:
            print('sigma=%g, log_likelihood=%.2f' % (sigma, log_likelihood))

    if verbose:
        print('====================')
        print('Best log_likelihood=%.2f\n' % best_log_likelihood)
    return best_log_likelihood

def kde_evaluation_tfd(test_data, samples, sigma_range=np.arange(0.05, 0.25, 0.01), verbose=True):
    return kde_evaluation(test_data, samples, sigma_range, verbose)

def kde_evaluation_all_folds(test_data, samples, sigma_range=np.arange(0.05, 0.25, 0.01), verbose=True):
    n_folds = len(samples)
    best_log_likelihood = float('-inf')
    for sigma in sigma_range:
        log_likelihood = [KDE(samples[i], sigma).average_log_likelihood(test_data[i]) for i in range(n_folds)]
        avg_log_likelihood = sum(log_likelihood) / float(n_folds)
        if avg_log_likelihood > best_log_likelihood:
            best_log_likelihood = avg_log_likelihood
        if verbose:
            print('sigma=%5g, log_likelihood=%8.2f   [%s]' %
                (sigma, avg_log_likelihood, ', '.join(['%8.2f' % l for l in log_likelihood])))

    if verbose:
        print('====================')
        print('Best log_likelihood=%.2f\n' % best_log_likelihood)
    return best_log_likelihood

def kde_eval_mnist(ae, armdn, test_data_loader, n_samples=10000, batch_size=10, sigma_range=np.arange(0.1, 1.00, 0.1), verbose=False):
# def kde_eval_mnist(valid_data_loader, test_data_loader, n_samples=10000, batch_size=100, sigma_range=np.arange(0.1, 0.30, 0.01), verbose=False):
    s_list = list()
    sampling_pbar = tqdm(range(int(n_samples / batch_size)))
    for i in sampling_pbar:
        z_ = armdn.sample(batch_size, 1.0)
        s = ae.forward(z_, forward_type='decoder')
        s_list.append(s.detach())
        # s_list.append(s.detach().cpu().numpy())
    # s = np.concatenate(s_list, axis=0)
    s = torch.cat(s_list, dim=0)

    # s_list = list()
    # for i, batch in enumerate(valid_data_loader):
    #     if i == 100:
    #         break
    #     s_list.append(batch['image'].cuda())
    #     # s_list.append(batch['image'].detach().numpy())
    # s = torch.cat(s_list, dim=0)

    test_data_list = list()
    for batch in test_data_loader:
        test_data_list.append(batch['image'])
        # test_data_list.append(batch['image'].detach().numpy())
    # test_data = np.concatenate(test_data_list, axis=0)
    test_data = torch.cat(test_data_list, dim=0)

    # s = np.reshape(s, (s.shape[0], -1))
    # test_data = np.reshape(test_data, (test_data.shape[0], -1))
    s = torch.reshape(s, (s.shape[0], -1))
    test_data = torch.reshape(test_data, (test_data.shape[0], -1))
    # s = s * 255
    # test_data = test_data * 255

    best_log_likelihood = float('-inf')
    best_se = 0
    best_sigma = 0

    # import parzen_theano
    sigma_range_pbar = tqdm(sigma_range)
    for sigma in sigma_range_pbar:
        # kde = parzen_theano.ParzenWindows(s, sigma)
        kde = parzen.ParzenWindows(s, sigma)
        log_likelihoods = kde.get_ll(test_data, batch_size=batch_size)
        ll = log_likelihoods.mean()
        se = log_likelihoods.std() / np.sqrt(test_data.shape[0])

        # kde = TorchAlternativeKDE(s, sigma)
        # ll, se = kde.average_se_log_likelihood(test_data, batch_size)
        if ll > best_log_likelihood:
            best_log_likelihood = ll
            best_se =  se
            best_sigma = sigma
        if verbose:
            print('sigma=%g, log_likelihood=%.2f (%.2f)' % (sigma, ll, se))

    if verbose:
        print('====================')
        print('Best log_likelihood=%.2f (%.2f)\n' % (best_log_likelihood, best_se))
    return best_log_likelihood, best_se, best_sigma

def kde_eval(test_data_loader, gen_img_paths, batch_size=10,
            sigma_range=np.arange(0.1, 1.00, 0.1), verbose=False):
    s_list = list()
    for img_path in gen_img_paths:
        img = scipy.misc.imread(img_path, mode='L')
        img = np.reshape(img, (1, img.shape[0], img.shape[1]))
        img = img / 255
        img = torch.from_numpy(img).float().cuda()
        img.requires_grad_(False)
        s_list.append(img)
    s = torch.cat(s_list, dim=0)

    test_data_list = list()
    for batch in test_data_loader:
        # print('test')
        test_data_list.append(batch['image'])
    test_data = torch.cat(test_data_list, dim=0)

    # s = np.reshape(s, (s.shape[0], -1))
    # test_data = np.reshape(test_data, (test_data.shape[0], -1))
    s = torch.reshape(s, (s.shape[0], -1))
    test_data = torch.reshape(test_data, (test_data.shape[0], -1))
    # s = s * 255
    # test_data = test_data * 255

    best_log_likelihood = float('-inf')
    best_se = 0
    best_sigma = 0

    # import parzen_theano
    sigma_range_pbar = tqdm(sigma_range)
    for sigma in sigma_range_pbar:
        # kde = parzen_theano.ParzenWindows(s, sigma)
        kde = parzen.ParzenWindows(s, sigma)
        log_likelihoods = kde.get_ll(test_data, batch_size=batch_size)
        ll = log_likelihoods.mean()
        se = log_likelihoods.std() / np.sqrt(test_data.shape[0])

        # kde = TorchAlternativeKDE(s, sigma)
        # ll, se = kde.average_se_log_likelihood(test_data, batch_size)
        if ll > best_log_likelihood:
            best_log_likelihood = ll
            best_se =  se
            best_sigma = sigma
        if verbose:
            print('sigma=%g, log_likelihood=%.2f (%.2f)' % (sigma, ll, se))

    if verbose:
        print('====================')
        print('Best log_likelihood=%.2f (%.2f)\n' % (best_log_likelihood, best_se))
    return best_log_likelihood, best_se, best_sigma


def kde_eval_tfd(ae, armdn, test_data_all_folds, n_samples=10000, batch_size=1000, sigma_range=np.arange(0.001, 0.200, 0.005), verbose=True):
    # s = net.generate_samples(n_samples=n_samples)
    s_list = list()
    for i in range(n_samples):
        z_ = armdn.sample(batch_size, 1.0)
        s = ae.forward(z_, forward_type='decoder')
        s_list.append(s.detach().cpu().numpy())
    s = np.concatenate(s_list, axis=0)

    best_log_likelihood = float('-inf')
    n_folds = len(test_data_all_folds)
    for sigma in sigma_range:
        kde = KDE(s, sigma)
        log_likelihood = [kde.average_log_likelihood(test_data_all_folds[i]) for i in range(n_folds)]
        avg_log_likelihood = sum(log_likelihood) / float(n_folds)
        avg_se = np.array(log_likelihood).std() / np.sqrt(n_folds)
        if avg_log_likelihood > best_log_likelihood:
            best_log_likelihood = avg_log_likelihood
            best_se = avg_se
            best_sigma = sigma
        if verbose:
            print('sigma=%5g, log_likelihood=%8.2f (%.2f)  [%s]' % \
                (sigma, avg_log_likelihood, avg_se, ', '.join(['%8.2f' % l for l in log_likelihood])))

    if verbose:
        print('====================')
        print('Best log_likelihood=%.2f (%.2f)\n' % (best_log_likelihood, best_se))
    return best_log_likelihood, best_se, best_sigma
