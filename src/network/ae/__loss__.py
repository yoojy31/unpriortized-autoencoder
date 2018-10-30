import torch
import torchvision

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg19_features = torchvision.models.vgg19_bn(pretrained=True).features
        for param in vgg19_features.parameters():
            param.requires_grad = False

        # VGG19
        # self.vgg19_relu1_1 = vgg19_features[:2]
        # self.vgg19_relu2_1 = vgg19_features[2:7]
        # self.vgg19_relu3_1 = vgg19_features[7:11]
        # VGG19-BN
        self.vgg19_relu1_1 = vgg19_features[:3]
        self.vgg19_relu2_1 = vgg19_features[3:10]
        self.vgg19_relu3_1 = vgg19_features[10:17]

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
        self.mean = torch.reshape(self.mean, (1, 3, 1, 1))
        self.std = torch.reshape(self.std, (1, 3, 1, 1))
        self.mean.requires_grad_(False)
        self.std.requires_grad_(False)

    def forward(self, _x, x):
        # the range of _x and x is [-1, 1]
        self.vgg19_relu1_1.train(mode=False)
        self.vgg19_relu2_1.train(mode=False)
        self.vgg19_relu3_1.train(mode=False)

        x.detach_()
        x.requires_grad_(False)

        _x = (_x + 1) * 0.5
        _x = (_x - self.mean) / self.std
        x = (x + 1) * 0.5
        x = (x - self.mean) / self.std

        x_relu1_1 = self.vgg19_relu1_1.forward(x)
        x_relu2_1 = self.vgg19_relu2_1.forward(x_relu1_1)
        x_relu3_1 = self.vgg19_relu3_1.forward(x_relu2_1)
        _x_relu1_1 = self.vgg19_relu1_1.forward(_x)
        _x_relu2_1 = self.vgg19_relu2_1.forward(_x_relu1_1)
        _x_relu3_1 = self.vgg19_relu3_1.forward(_x_relu2_1)

        relu1_1_mse = torch.nn.functional.mse_loss(_x_relu1_1, x_relu1_1)
        relu2_1_mse = torch.nn.functional.mse_loss(_x_relu2_1, x_relu2_1)
        relu3_1_mse = torch.nn.functional.mse_loss(_x_relu3_1, x_relu3_1)
        perc_loss = relu1_1_mse + 10 * relu2_1_mse + 100 * relu3_1_mse
        # perc_loss = 0.5 * (relu1_1_mse + relu2_1_mse + relu3_1_mse)
        # VGG19: tensor(0.3828, device='cuda:1') tensor(3.3571, device='cuda:1') tensor(70.6274, device='cuda:1')
        # VGG19-BN: tensor(0.022634, device='cuda:1') tensor(0.022329, device='cuda:1') tensor(0.023558, device='cuda:1')
        return perc_loss


# VGG19-BN
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace)
#   (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU(inplace)
#   (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (9): ReLU(inplace)
#   (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (12): ReLU(inplace)
#   (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (16): ReLU(inplace)
#   (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (19): ReLU(inplace)
#   (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (22): ReLU(inplace)
#   (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (25): ReLU(inplace)
#   (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (29): ReLU(inplace)
#   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (32): ReLU(inplace)
#   (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (35): ReLU(inplace)
#   (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (38): ReLU(inplace)
#   (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (42): ReLU(inplace)
#   (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (45): ReLU(inplace)
#   (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (48): ReLU(inplace)
#   (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (51): ReLU(inplace)
#   (52): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
