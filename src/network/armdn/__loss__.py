import math
import torch

def mdn_loss_fn(z, mu, sig, pi):
    p_gauss = gaussian_pdf(z, mu, sig)
    # print(p_gauss.shape)
    result = pi * p_gauss
    result = torch.sum(result, dim=2)
    result = -torch.log(result + 1e-12)
    return torch.mean(result)

def gaussian_pdf(z, mu, sig):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (z.expand_as(mu) - mu) / sig
    result = -0.5 * (result * result)
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result
