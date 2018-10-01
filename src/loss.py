import math
import torch
import torchvision

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg19_features = torchvision.models.vgg19(pretrained=True).features
        self.vgg19_relu1_1 = vgg19_features[:2]
        self.vgg19_relu2_1 = vgg19_features[2:7]
        self.vgg19_relu3_1 = vgg19_features[7:11]

    def forward(self, _x, x):
        x.requires_grad_(True)
        _x.requires_grad_(True)

        x_relu1_1 = self.vgg19_relu1_1.forward(x)
        x_relu2_1 = self.vgg19_relu2_1.forward(x_relu1_1)
        x_relu3_1 = self.vgg19_relu3_1.forward(x_relu2_1)
        _x_relu1_1 = self.vgg19_relu1_1.forward(_x)
        _x_relu2_1 = self.vgg19_relu2_1.forward(_x_relu1_1)
        _x_relu3_1 = self.vgg19_relu3_1.forward(_x_relu2_1)

        relu1_1_mse = torch.nn.functional.mse_loss(_x_relu1_1, x_relu1_1)
        relu2_1_mse = torch.nn.functional.mse_loss(_x_relu2_1, x_relu2_1)
        relu3_1_mse = torch.nn.functional.mse_loss(_x_relu3_1, x_relu3_1)
        perc_mse = 0.5 * relu1_1_mse + relu2_1_mse + relu3_1_mse
        return perc_mse

def gaussian_pdf(z, mu, sig):
    # print(z.shape)
    # print(mu.shape)
    # print(sig.shape)
    # exit()
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (z.expand_as(mu) - mu) / sig
    result = -0.5 * (result * result)
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result

def mdn_loss_fn(z, mu, sig, pi):
    p_gauss = gaussian_pdf(z, mu, sig)
    # print(p_gauss.shape)
    result = pi * p_gauss
    result = torch.sum(result, dim=2)
    result = -torch.log(result + 1e-12)
    return torch.mean(result)
