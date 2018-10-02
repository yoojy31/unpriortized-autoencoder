import torchvision
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg19_features = torchvision.models.vgg19(pretrained=True).features
        for param in vgg19_features.parameters():
            param.requires_grad = False

        # VGG19
        self.vgg19_relu1_1 = vgg19_features[:2]
        self.vgg19_relu2_1 = vgg19_features[2:7]
        self.vgg19_relu3_1 = vgg19_features[7:11]

    def forward(self, _x, x):
        x.detach_()
        x.requires_grad_(False)

        x_relu1_1 = self.vgg19_relu1_1.forward(x)
        x_relu2_1 = self.vgg19_relu2_1.forward(x_relu1_1)
        x_relu3_1 = self.vgg19_relu3_1.forward(x_relu2_1)
        _x_relu1_1 = self.vgg19_relu1_1.forward(_x)
        _x_relu2_1 = self.vgg19_relu2_1.forward(_x_relu1_1)
        _x_relu3_1 = self.vgg19_relu3_1.forward(_x_relu2_1)

        relu1_1_mse = nn.functional.mse_loss(_x_relu1_1, x_relu1_1)
        relu2_1_mse = nn.functional.mse_loss(_x_relu2_1, x_relu2_1)
        relu3_1_mse = nn.functional.mse_loss(_x_relu3_1, x_relu3_1)
        perc_loss = 0.5 * (relu1_1_mse + relu2_1_mse + relu3_1_mse)
        return perc_loss
