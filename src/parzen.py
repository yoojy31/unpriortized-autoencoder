"""
This file was copyed from pyleran2.distributions.parzen.py
Their license is BSD clause-3: https://github.com/lisa-lab/pylearn2/
"""
import math
import numpy as np
# import theano
import torch
# T = theano.tensor


class ParzenWindows(object):
    """
    Parzen Window estimation and log-likelihood calculator.
    This is usually used to test generative models as follows:
    1 - Get 10k samples from the generative model
    2 - Contruct a ParzenWindows object with the samples from 1
    3 - Test the log-likelihood on the test set
    Parameters
    ----------
    samples : numpy matrix
        See description for make_lpdf
    sigma : scalar
        See description for make_lpdf
    """
    def __init__(self, samples, sigma):
        # just keeping these for debugging/examination, not needed
        self._samples = samples
        self._sigma = sigma

        # self.lpdf = parzen_window(samples, sigma)

    def get_ll(self, x, batch_size=100):
        """
        Evaluates the log likelihood of a set of datapoints with respect to the
        probability distribution.
        Parameters
        ----------
        x : numpy matrix
            The set of points for which you want to evaluate the log \
            likelihood.
        """
        inds = range(x.shape[0])
        n_batches = int(math.ceil(float(len(inds)) / batch_size))

        lls = []
        for i in range(n_batches):
            ll_batch = self.calc_ll(x[inds[i::n_batches]].cuda())
            lls.extend(ll_batch.detach().cpu().numpy())
        # return numpy.array(lls).mean()
        # lls = torch.cat(lls, dim=0)
        return np.array(lls)#.mean()

    def calc_ll(self, x):
        """
        Makes a Theano function that allows the evalution of a Parzen windows
        estimator (aka kernel density estimator) where the Kernel is a normal
        distribution with stddev sigma and with points at mu.
        Parameters
        -----------
        mu : numpy matrix
            Contains the data points over which this distribution is based.
        sigma : scalar
            The standard deviation of the normal distribution around each data
            point.
        Returns
        -------
        lpdf : callable
            Estimator of the log of the probability density under a point.
        """

        # x_max = x.max(axis=axis)
        # return (x_max + np.log(np.exp(x - x_max[:,np.newaxis]).sum(axis=axis)))# .asarray()
        def log_mean_exp(a):
            """
            We need the log-likelihood, this calculates the logarithm
            of a Parzen window

            This computes log(exp(x_1) + exp(x_2) + ... + exp(x_n)) as
            x* + log(exp(x_1-x*) + exp(x_2-x*) + ... + exp(x_n-x*)), where x* is the
            max over all x_i.  This can avoid numerical problems.
            """
            max_, _ = a.max(dim=1)
            max__ = torch.reshape(max_, (-1, 1))
            # print(a.shape, max_.shape, max__.shape)
            # exit()

            # print(a.shape, max_.shape)
            # return max_ + torch.log( torch.exp(a - max_.dimshuffle(0, 'x')).mean(1))
            # print(torch.exp(a - max_).mean(dim=1).shape)
            # return max_ + torch.log(torch.sum(torch.exp(a - max__), dim=1))
            return max_ + torch.log(torch.mean(torch.exp(a - max__), dim=1))

        s = self._samples
        # x: (100, 784)
        # mu(s): (10000, 784)
        # x = T.matrix()
        # mu(s) = theano.shared(mu)

        # print('sampling:', s.shape, 'test:', x.shape)
        # x: (100, 1, 784)
        # mu(s_): (1, 10000, 784)
        # a = (x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1)) / sigma
        x_ = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        shape = s.shape
        s_ = torch.reshape(s, (1, shape[0], shape[1]))

        # a: (100, 10000, 784)
        # E: (100)
        a = (x_ - s_) / self._sigma
        E = log_mean_exp(-0.5 * torch.sum(a**2, dim=2))
        Z = s.shape[1] * math.log(self._sigma * math.sqrt(math.pi * 2))
        result = E - Z

        # # a: (100, 10000, 784)
        # # a = -0.5 * (( (x_ - s_) / self._sigma) ** 2)
        # a = -0.5 * torch.sum(( (x_ - s_) / self._sigma) ** 2, dim=2, keepdim=True)
        # # a = -0.5 * (( torch.norm(x_ - s_, p=1, dim=2, keepdim=True) / self._sigma) ** 2)
        # # a = -0.5 * (( torch.norm(x_ - s_, p=2, dim=2, keepdim=True) / self._sigma) ** 2)
        # a_max, _ = a.max(dim=1, keepdim=True)
        # E = a_max + torch.log(torch.exp(a - a_max).sum(dim=1, keepdim=True)) - math.log(a.shape[1])
        # Z = 0.5 * a.shape[2] * math.log(2 * math.pi * (self._sigma ** 2)) # (2πh^2)^(D/2)
        # result = E - Z
        # # result = torch.sum(E - Z, dim=2)
        # result = torch.squeeze(result)
        # # # return theano.function([x], E - Z)
        return result
