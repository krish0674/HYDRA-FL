import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models.RayAdaIN import HyperRayAdaBN


class InitialBlockHN(nn.Module):
    def __init__(self, in_channels=5, out_channels=13, bn_mode=None):
        super().__init__()

        self.bn_mode = bn_mode
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.conv = nn.Conv2d(in_channels,
        #                       out_channels,
        #                       kernel_size=3,
        #                       stride=2,
        #                       padding=1)

        self.prelu = nn.ReLU()

        if self.bn_mode == "bn":
            self.batchnorm = nn.BatchNorm2d(out_channels)
        elif self.bn_mode == "hbn":
            self.batchnorm = HyperRayAdaBN(out_channels)

    def forward(self, x, weights):
        # main = self.conv(x)

        main = F.conv2d(
            x,
            weight=weights["conv0_weights"],
            bias=weights["conv0_bias"],
            stride=2,
            padding=1,
        )

        if self.bn_mode == "bn":
            x = self.batchnorm(x)
        elif self.bn_mode == "hbn":
            main = self.batchnorm(
                main, scale=weights["bn_0_scale"], translate=weights["bn_0_translate"]
            )

        side = self.maxpool(x)

        x = torch.cat((main, side), dim=1)
        x = self.prelu(x)

        return x


class InitialBlock(nn.Module):
    def __init__(self, in_channels=5, out_channels=13):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

        self.prelu = nn.PReLU(18)

        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        main = self.conv(x)
        main = self.batchnorm(main)

        side = self.maxpool(x)

        x = torch.cat((main, side), dim=1)
        x = self.prelu(x)

        return x


class InitialBlockVanila(nn.Module):
    def __init__(self, in_channels=5, out_channels=13):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

        self.prelu = nn.PReLU(in_channels+out_channels)

        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.conv_1x1 = nn.Conv2d(in_channels + out_channels, 18, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        main = self.conv(x)
        main = self.batchnorm(main)

        side = self.maxpool(x)

        x = torch.cat((main, side), dim=1)
        x = self.prelu(x)
        x = self.conv_1x1(x)

        return x
