import math
import torch

# source: https://github.com/jmtomczak/vae_vampprior/blob/master/utils/distributions.py
def log_gaussian_pdf(x, _x, log_var, average=True, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - _x, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

# source: https://github.com/jmtomczak/vae_vampprior/blob/master/utils/distributions.py
min_epsilon = 1e-5
max_epsilon = 1.-1e-5
def log_bernoulli_pdf(x, _x, average=True, dim=None):
    probs = torch.clamp( _x, min=min_epsilon, max=max_epsilon )
    log_bernoulli = x * torch.log( probs ) + (1. - x ) * torch.log( 1. - probs )
    if average:
        return torch.mean( log_bernoulli, dim )
    else:
        return torch.sum( log_bernoulli, dim )

def gaussian_pdf(x, mu, sig):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (x.expand_as(mu) - mu) / sig
    result = -0.5 * (result * result)
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result
