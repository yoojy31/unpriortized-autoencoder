from .armdn01 import ARMDN01

class ARMDN02(ARMDN01):
    def forward(self, *x):
        self.train(mode=True)
        result = super(ARMDN02, self).forward(*x)
        return result

    def sample(self, n_sample, tau):
        self.train(mode=True)
        result = super(ARMDN02, self).sample(n_sample, tau)
        return result
