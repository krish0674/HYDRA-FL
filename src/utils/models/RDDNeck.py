import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models.RayAdaIN import RayAdaIN, RayAdaBN, HyperRayAdaBN


class RDDNeckHN(nn.Module):
    def __init__(
        self,
        dilation,
        in_channels,
        out_channels,
        down_flag,
        device,
        relu=True,
        projection_ratio=4,
        p=0.1,
        bn_mode=None,
    ):
        super().__init__()

        # Define class variables
        self.bn_mode = bn_mode
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.dilation = dilation
        self.down_flag = down_flag

        self.device = device

        if down_flag:
            self.stride = 2
            self.reduced_depth = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.reduced_depth = int(out_channels // projection_ratio)

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, return_indices=True
        )

        self.dropout = nn.Dropout2d(p=p)

        # self.conv1 = nn.Conv2d(in_channels=self.in_channels,
        #                        out_channels=self.reduced_depth,
        #                        kernel_size=1,
        #                        stride=1,
        #                        padding=0,
        #                        bias=False,
        #                        dilation=1)

        self.prelu1 = activation

        # self.conv2 = nn.Conv2d(in_channels=self.reduced_depth,
        #                        out_channels=self.reduced_depth,
        #                        kernel_size=3,
        #                        stride=self.stride,
        #                        padding=self.dilation,
        #                        bias=True,
        #                        dilation=self.dilation)

        self.prelu2 = activation

        # self.conv3 = nn.Conv2d(in_channels=self.reduced_depth,
        #                        out_channels=self.out_channels,
        #                        kernel_size=1,
        #                        stride=1,
        #                        padding=0,
        #                        bias=False,
        #                        dilation=1)

        self.prelu3 = activation

        if self.bn_mode == "bn":
            self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
            self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
            self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
        elif self.bn_mode == "hbn":
            self.batchnorm1 = HyperRayAdaBN(self.reduced_depth)
            self.batchnorm2 = HyperRayAdaBN(self.reduced_depth)
            self.batchnorm3 = HyperRayAdaBN(self.out_channels)

    def forward(self, x, weights):
        bs = x.size()[0]
        x_copy = x

        # Side Branch
        # x = self.conv1(x)
        x = F.conv2d(
            x, weight=weights["conv_0_weights"], stride=1, padding=0, dilation=1
        )

        if self.bn_mode == "bn":
            x = self.batchnorm1(x)
        elif self.bn_mode == "hbn":
            x = self.batchnorm1(
                x, scale=weights["bn_0_scale"], translate=weights["bn_0_translate"]
            )
        x = self.prelu1(x)

        # x = self.conv2(x)
        x = F.conv2d(
            x,
            weight=weights["conv_1_weights"],
            bias=weights["conv_1_bias"],
            stride=self.stride,
            padding=self.dilation,
            dilation=self.dilation,
        )

        if self.bn_mode == "bn":
            x = self.batchnorm2(x)
        elif self.bn_mode == "hbn":
            x = self.batchnorm2(
                x, scale=weights["bn_1_scale"], translate=weights["bn_1_translate"]
            )

        x = self.prelu2(x)

        # x = self.conv3(x)
        x = F.conv2d(
            x, weight=weights["conv_2_weights"], stride=1, padding=0, dilation=1
        )

        if self.bn_mode == "bn":
            x = self.batchnorm3(x)
        elif self.bn_mode == "hbn":
            x = self.batchnorm3(
                x, scale=weights["bn_2_scale"], translate=weights["bn_2_translate"]
            )

        x = self.dropout(x)

        # Main Branch
        if self.down_flag:
            x_copy, indices = self.maxpool(x_copy)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3])).to(
                self.device
            )
            # extras = extras.to(device)
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Sum of main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        if self.down_flag:
            return x, indices
        else:
            return x


class RDDNeck(nn.Module):
    def __init__(
        self,
        dilation,
        in_channels,
        out_channels,
        down_flag,
        device,
        relu=False,
        projection_ratio=4,
        p=0.1,
        ray_dim=None,
    ):
        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.device = device

        self.out_channels = out_channels
        self.dilation = dilation
        self.down_flag = down_flag
        self.ray_dim = ray_dim

        if down_flag:
            self.stride = 2
            self.reduced_depth = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.reduced_depth = int(out_channels // projection_ratio)

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, return_indices=True
        )

        self.dropout = nn.Dropout2d(p=p)

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.reduced_depth,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.prelu1 = activation

        self.conv2 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=3,
            stride=self.stride,
            padding=self.dilation,
            bias=True,
            dilation=self.dilation,
        )

        self.prelu2 = activation

        self.conv3 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.prelu3 = activation
        if self.ray_dim is None:
            self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
            self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
            self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
        else:
            self.batchnorm1 = RayAdaBN(self.ray_dim, 10, self.reduced_depth)
            self.batchnorm2 = RayAdaBN(self.ray_dim, 10, self.reduced_depth)
            self.batchnorm3 = RayAdaBN(self.ray_dim, 10, self.out_channels)

    def forward(self, x, ray=None):
        bs = x.size()[0]
        x_copy = x

        # Side Branch
        x = self.conv1(x)
        if ray is None:
            x = self.batchnorm1(x)
        else:
            x = self.batchnorm1(x, ray)
        x = self.prelu1(x)

        x = self.conv2(x)
        if ray is None:
            x = self.batchnorm2(x)
        else:
            x = self.batchnorm2(x, ray)
        x = self.prelu2(x)

        x = self.conv3(x)
        if ray is None:
            x = self.batchnorm3(x)
        else:
            x = self.batchnorm3(x, ray)

        x = self.dropout(x)

        # Main Branch
        if self.down_flag:
            x_copy, indices = self.maxpool(x_copy)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
            extras = extras.to(self.device)
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Sum of main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        if self.down_flag:
            return x, indices
        else:
            return x


class RDDNeckVanila(nn.Module):
    def __init__(
        self,
        dilation,
        in_channels,
        out_channels,
        down_flag,
        device,
        relu=False,
        projection_ratio=4,
        p=0.1,
    ):
        super().__init__()

        # Define class variables
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.dilation = dilation
        self.down_flag = down_flag
        self.device = device

        if down_flag:
            self.stride = 2
            self.reduced_depth = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.reduced_depth = int(out_channels // projection_ratio)

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, return_indices=True
        )

        self.dropout = nn.Dropout2d(p=p)

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.reduced_depth,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.prelu1 = activation

        self.conv2 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=3,
            stride=self.stride,
            padding=self.dilation,
            bias=True,
            dilation=self.dilation,
        )

        self.prelu2 = activation

        self.conv3 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.prelu3 = activation

        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        bs = x.size()[0]
        x_copy = x

        # Side Branch
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        x = self.dropout(x)

        # Main Branch
        if self.down_flag:
            x_copy, indices = self.maxpool(x_copy)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
            extras = extras.to(self.device)
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Sum of main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        if self.down_flag:
            return x, indices
        else:
            return x
