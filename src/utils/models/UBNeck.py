import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models.RayAdaIN import HyperRayAdaBN


class UBNeckHN(nn.Module):
    def __init__(
        self, in_channels, out_channels, relu=True, projection_ratio=4, bn_mode=None
    ):
        super().__init__()

        # Define class variables
        self.bn_mode = bn_mode
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # self.main_conv = nn.Conv2d(in_channels=self.in_channels,
        #                            out_channels=self.out_channels,
        #                            kernel_size=1)

        self.dropout = nn.Dropout2d(p=0.1)

        # self.convt1 = nn.ConvTranspose2d(in_channels=self.in_channels,
        #                                  out_channels=self.reduced_depth,
        #                                  kernel_size=1,
        #                                  padding=0,
        #                                  bias=False)

        self.prelu1 = activation

        # self.convt2 = nn.ConvTranspose2d(in_channels=self.reduced_depth,
        #                                  out_channels=self.reduced_depth,
        #                                  kernel_size=3,
        #                                  stride=2,
        #                                  padding=1,
        #                                  output_padding=1,
        #                                  bias=False)

        self.prelu2 = activation

        # self.convt3 = nn.ConvTranspose2d(in_channels=self.reduced_depth,
        #                                  out_channels=self.out_channels,
        #                                  kernel_size=1,
        #                                  padding=0,
        #                                  bias=False)

        self.prelu3 = activation

        if self.bn_mode == "bn":
            self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
            self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
            self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
        elif self.bn_mode == "hbn":
            self.batchnorm1 = HyperRayAdaBN(self.reduced_depth)
            self.batchnorm2 = HyperRayAdaBN(self.reduced_depth)
            self.batchnorm3 = HyperRayAdaBN(self.out_channels)

    def forward(self, x, weights, indices):
        x_copy = x

        # Side Branch
        # x = self.convt1(x)
        x = F.conv_transpose2d(x, weight=weights["convt_0_weights"])

        if self.bn_mode == "bn":
            x = self.batchnorm1(x)
        elif self.bn_mode == "hbn":
            x = self.batchnorm1(
                x, scale=weights["bn_0_scale"], translate=weights["bn_0_translate"]
            )

        x = self.prelu1(x)

        # x = self.convt2(x)
        x = F.conv_transpose2d(
            x, weight=weights["convt_1_weights"], stride=2, padding=1, output_padding=1
        )

        if self.bn_mode == "bn":
            x = self.batchnorm2(x)
        elif self.bn_mode == "hbn":
            x = self.batchnorm2(
                x, scale=weights["bn_1_scale"], translate=weights["bn_1_translate"]
            )

        x = self.prelu2(x)

        # x = self.convt3(x)
        x = F.conv_transpose2d(x, weight=weights["convt_2_weights"])

        if self.bn_mode == "bn":
            x = self.batchnorm3(x)
        elif self.bn_mode == "hbn":
            x = self.batchnorm3(
                x, scale=weights["bn_2_scale"], translate=weights["bn_2_translate"]
            )

        x = self.dropout(x)

        # Main Branch

        # x_copy = self.main_conv(x_copy)
        x_copy = F.conv2d(
            x_copy, weight=weights["conv_0_weights"], bias=weights["conv_0_bias"]
        )
        x_copy = self.unpool(x_copy, indices, output_size=x.size())

        # Concat
        x = x + x_copy
        x = self.prelu3(x)

        return x


class UBNeck(nn.Module):
    def __init__(self, in_channels, out_channels, relu=False, projection_ratio=4):
        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.main_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1
        )

        self.dropout = nn.Dropout2d(p=0.1)

        self.convt1 = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.reduced_depth,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.prelu1 = activation

        self.convt2 = nn.ConvTranspose2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

        self.prelu2 = activation

        self.convt3 = nn.ConvTranspose2d(
            in_channels=self.reduced_depth,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.prelu3 = activation

        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x, indices):
        x_copy = x

        # Side Branch
        x = self.convt1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)

        x = self.convt2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)

        x = self.convt3(x)
        x = self.batchnorm3(x)

        x = self.dropout(x)

        # Main Branch

        x_copy = self.main_conv(x_copy)
        x_copy = self.unpool(x_copy, indices, output_size=x.size())

        # Concat
        x = x + x_copy
        x = self.prelu3(x)

        return x


class UBNeckVanila(nn.Module):
    def __init__(self, in_channels, out_channels, relu=False, projection_ratio=4):
        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.main_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1
        )

        self.dropout = nn.Dropout2d(p=0.1)

        self.convt1 = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.reduced_depth,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.prelu1 = activation

        self.convt2 = nn.ConvTranspose2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

        self.prelu2 = activation

        self.convt3 = nn.ConvTranspose2d(
            in_channels=self.reduced_depth,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.prelu3 = activation

        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x, indices):
        x_copy = x

        # Side Branch
        x = self.convt1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)

        x = self.convt2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)

        x = self.convt3(x)
        x = self.batchnorm3(x)

        x = self.dropout(x)

        # Main Branch

        x_copy = self.main_conv(x_copy)
        x_copy = self.unpool(x_copy, indices, output_size=x.size())

        # Concat
        x = x + x_copy
        x = self.prelu3(x)

        return x
