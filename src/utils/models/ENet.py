import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models.InitialBlock import InitialBlockHN, InitialBlock, InitialBlockVanila
from utils.models.RDDNeck import RDDNeckHN, RDDNeck, RDDNeckVanila
from utils.models.UBNeck import UBNeckHN, UBNeck, UBNeckVanila
from utils.models.ASNeck import ASNeckHN, ASNeck, ASNeckVanila
from utils.models.channels import rddn_channels, ubn_channels

from collections import OrderedDict

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class ENetTarget(nn.Module):
    def __init__(self, n_class, device):
        super().__init__()

        # Define class variables
        self.n_class = n_class

        self.device = device

        ####################
        #                  #
        #     ENCODER      #
        #                  #
        ####################

        # The initial block
        self.init = InitialBlockHN(bn_mode="hbn")

        # The first bottleneck
        self.rdnn_0 = RDDNeckHN(
            dilation=1,
            in_channels=18,
            out_channels=72,
            down_flag=True,
            device=self.device,
            p=0.01,
            bn_mode="hbn",
        )

        self.rdnn_1 = RDDNeckHN(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            device=self.device,
            p=0.01,
            bn_mode="bn",
        )

        self.rdnn_2 = RDDNeckHN(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            device=self.device,
            p=0.01,
            bn_mode="hbn",
        )

        self.rdnn_3 = RDDNeckHN(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            device=self.device,
            p=0.01,
            bn_mode="bn",
        )

        self.rdnn_4 = RDDNeckHN(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            device=self.device,
            p=0.01,
            bn_mode="hbn",
        )

        # The second bottleneck
        self.rdnn_5 = RDDNeckHN(
            dilation=1,
            in_channels=72,
            out_channels=144,
            down_flag=True,
            device=self.device,
            bn_mode="bn",
        )

        self.rdnn_6 = RDDNeckHN(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="hbn",
        )

        self.rdnn_7 = RDDNeckHN(
            dilation=2,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="bn",
        )

        self.asn_0 = ASNeckHN(in_channels=144, out_channels=144, bn_mode="hbn")

        self.rdnn_8 = RDDNeckHN(
            dilation=4,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="hbn",
        )

        self.rdnn_9 = RDDNeckHN(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="bn",
        )

        self.rdnn_10 = RDDNeckHN(
            dilation=8,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="hbn",
        )

        self.asn_1 = ASNeckHN(in_channels=144, out_channels=144, bn_mode="bn")

        self.rdnn_11 = RDDNeckHN(
            dilation=16,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="bn",
        )

        ####################
        #                  #
        #     DECODER      #
        #                  #
        ####################

        # The third bottleneck
        self.rdnn_12 = RDDNeckHN(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="hbn",
        )

        self.rdnn_13 = RDDNeckHN(
            dilation=2,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="bn",
        )

        self.asn_2 = ASNeckHN(in_channels=144, out_channels=144, bn_mode="hbn")

        self.rdnn_14 = RDDNeckHN(
            dilation=4,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="hbn",
        )

        self.rdnn_15 = RDDNeckHN(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="bn",
        )

        self.rdnn_16 = RDDNeckHN(
            dilation=8,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="hbn",
        )

        self.asn_3 = ASNeckHN(in_channels=144, out_channels=144, bn_mode="bn")

        self.rdnn_17 = RDDNeckHN(
            dilation=16,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            bn_mode="bn",
        )

        # The fourth bottleneck
        self.ubn_0 = UBNeckHN(
            in_channels=144, out_channels=72, relu=True, bn_mode="hbn"
        )

        self.rdnn_18 = RDDNeckHN(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            device=self.device,
            relu=True,
            bn_mode="hbn",
        )

        self.rdnn_19 = RDDNeckHN(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            device=self.device,
            relu=True,
            bn_mode="bn",
        )

        # The fifth bottleneck
        self.ubn_1 = UBNeckHN(in_channels=72, out_channels=18, relu=True, bn_mode="bn")

        self.rdnn_20 = RDDNeckHN(
            dilation=1,
            in_channels=18,
            out_channels=18,
            down_flag=False,
            device=self.device,
            relu=True,
            bn_mode="hbn",
        )

        # Final ConvTranspose Layer
        # self.fullconv = nn.ConvTranspose2d(in_channels=16,
        #                                    out_channels=self.n_class,
        #                                    kernel_size=3,
        #                                    stride=2,
        #                                    padding=1,
        #                                    output_padding=1,
        #                                    bias=False)

    def forward(self, x, weights):
        # The initial block
        x = self.init(x, weights["initial_block"])

        # The first bottleneck
        x, i1 = self.rdnn_0(x, weights["rddn_0"])
        x = self.rdnn_1(x, weights["rddn_1"])
        x = self.rdnn_2(x, weights["rddn_2"])
        x = self.rdnn_3(x, weights["rddn_3"])
        x = self.rdnn_4(x, weights["rddn_4"])

        # The second bottleneck
        x, i2 = self.rdnn_5(x, weights["rddn_5"])
        x = self.rdnn_6(x, weights["rddn_6"])
        x = self.rdnn_7(x, weights["rddn_7"])
        x = self.asn_0(x, weights["asn_0"])
        x = self.rdnn_8(x, weights["rddn_8"])
        x = self.rdnn_9(x, weights["rddn_9"])
        x = self.rdnn_10(x, weights["rddn_10"])
        x = self.asn_1(x, weights["asn_1"])
        x = self.rdnn_11(x, weights["rddn_11"])

        # The third bottleneck
        x = self.rdnn_12(x, weights["rddn_12"])
        x = self.rdnn_13(x, weights["rddn_13"])
        x = self.asn_2(x, weights["asn_2"])
        x = self.rdnn_14(x, weights["rddn_14"])
        x = self.rdnn_15(x, weights["rddn_15"])
        x = self.rdnn_16(x, weights["rddn_16"])
        x = self.asn_3(x, weights["asn_3"])
        x = self.rdnn_17(x, weights["rddn_17"])

        # The fourth bottleneck
        x = self.ubn_0(x, weights["ubn_0"], i2)
        x = self.rdnn_18(x, weights["rddn_18"])
        x = self.rdnn_19(x, weights["rddn_19"])

        # The fifth bottleneck
        x = self.ubn_1(x, weights["ubn_1"], i1)
        x = self.rdnn_20(x, weights["rddn_20"])

        # Final ConvTranspose Layer
        # x = self.fullconv(x)
        task1_output = F.conv_transpose2d(
            x,
            weight=weights["final_convt"]["task_0_convt_weights"],
            stride=2,
            padding=1,
            output_padding=1,
        )
        task2_output = F.conv_transpose2d(
            x,
            weight=weights["final_convt"]["task_1_convt_weights"],
            stride=2,
            padding=1,
            output_padding=1,
        )

        return task1_output, task2_output


class EnetHyper(nn.Module):
    """Modification of https://github.com/pytorch/examples/blob/master/mnist/main.py"""

    def __init__(
            self,
            ray_hidden_dim=100,
            n_rddn=21,
            n_asn=4,
            n_ubn=2,
            n_classes=12,
            rddn_c=rddn_channels,
            ubn_c=ubn_channels,
            bn_mode="bn",
    ):
        super().__init__()

        self.bn_mode = bn_mode
        self.n_rddn = n_rddn
        self.n_asn = n_asn
        self.n_ubn = n_ubn
        self.rddn_c = rddn_c
        self.ubn_c = ubn_c
        assert len(self.rddn_c) == self.n_rddn, "mis-match number of rddn layers"
        assert len(self.ubn_c) == self.n_ubn, "mis-match number of ubn layers"
        self.n_classes = n_classes

        self.ray_mlp = nn.Sequential(
            nn.Linear(25, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.initial_block_weights = nn.Linear(ray_hidden_dim, 5 * 13 * 3 * 3)
        self.initial_block_bias = nn.Linear(ray_hidden_dim, 13)
        self.initial_block_scale = nn.Linear(ray_hidden_dim, 13)
        self.initial_block_translate = nn.Linear(ray_hidden_dim, 13)

        for i in range(self.n_rddn):
            in_c, mid_c, out_c = (
                self.rddn_c[i][0],
                self.rddn_c[i][0] // 4,
                self.rddn_c[i][1],
            )
            setattr(
                self,
                f"rddn_{i}_conv_0_weights",
                nn.Linear(ray_hidden_dim, mid_c * in_c * 1 * 1),
            )
            setattr(
                self,
                f"rddn_{i}_conv_1_weights",
                nn.Linear(ray_hidden_dim, mid_c * mid_c * 3 * 3),
            )
            setattr(self, f"rddn_{i}_conv_1_bias", nn.Linear(ray_hidden_dim, mid_c))
            setattr(
                self,
                f"rddn_{i}_conv_2_weights",
                nn.Linear(ray_hidden_dim, out_c * mid_c * 1 * 1),
            )
            if (self.bn_mode == "hbn") and (i % 2 == 0):
                setattr(self, f"rddn_{i}_bn_0_scale", nn.Linear(ray_hidden_dim, mid_c))
                setattr(
                    self, f"rddn_{i}_bn_0_translate", nn.Linear(ray_hidden_dim, mid_c)
                )
                setattr(self, f"rddn_{i}_bn_1_scale", nn.Linear(ray_hidden_dim, mid_c))
                setattr(
                    self, f"rddn_{i}_bn_1_translate", nn.Linear(ray_hidden_dim, mid_c)
                )
                setattr(self, f"rddn_{i}_bn_2_scale", nn.Linear(ray_hidden_dim, out_c))
                setattr(
                    self, f"rddn_{i}_bn_2_translate", nn.Linear(ray_hidden_dim, out_c)
                )

        for i in range(self.n_asn):
            setattr(
                self,
                f"asn_{i}_conv_0_weights",
                nn.Linear(ray_hidden_dim, 36 * 144 * 1 * 1),
            )
            setattr(
                self,
                f"asn_{i}_conv_11_weights",
                nn.Linear(ray_hidden_dim, 36 * 36 * 1 * 5),
            )
            setattr(
                self,
                f"asn_{i}_conv_12_weights",
                nn.Linear(ray_hidden_dim, 36 * 36 * 5 * 1),
            )
            setattr(
                self,
                f"asn_{i}_conv_2_weights",
                nn.Linear(ray_hidden_dim, 144 * 36 * 1 * 1),
            )
            if (self.bn_mode == "hbn") and (i % 2 == 0):
                setattr(self, f"asn_{i}_bn_0_scale", nn.Linear(ray_hidden_dim, 36))
                setattr(self, f"asn_{i}_bn_0_translate", nn.Linear(ray_hidden_dim, 36))
                setattr(self, f"asn_{i}_bn_1_scale", nn.Linear(ray_hidden_dim, 36))
                setattr(self, f"asn_{i}_bn_1_translate", nn.Linear(ray_hidden_dim, 36))
                setattr(self, f"asn_{i}_bn_2_scale", nn.Linear(ray_hidden_dim, 144))
                setattr(self, f"asn_{i}_bn_2_translate", nn.Linear(ray_hidden_dim, 144))

        for i in range(self.n_ubn):
            in_c, mid_c, out_c = (
                self.ubn_c[i][0],
                self.ubn_c[i][0] // 4,
                self.ubn_c[i][1],
            )
            setattr(
                self,
                f"ubn_{i}_convt_0_weights",
                nn.Linear(ray_hidden_dim, in_c * mid_c * 1 * 1),
            )
            setattr(
                self,
                f"ubn_{i}_convt_1_weights",
                nn.Linear(ray_hidden_dim, mid_c * mid_c * 3 * 3),
            )
            setattr(
                self,
                f"ubn_{i}_convt_2_weights",
                nn.Linear(ray_hidden_dim, out_c * mid_c * 1 * 1),
            )
            setattr(
                self,
                f"ubn_{i}_conv_0_weights",
                nn.Linear(ray_hidden_dim, out_c * in_c * 1 * 1),
            )
            setattr(self, f"ubn_{i}_conv_0_bias", nn.Linear(ray_hidden_dim, out_c))
            if (self.bn_mode == "hbn") and (i % 2 == 0):
                setattr(self, f"ubn_{i}_bn_0_scale", nn.Linear(ray_hidden_dim, mid_c))
                setattr(
                    self, f"ubn_{i}_bn_0_translate", nn.Linear(ray_hidden_dim, mid_c)
                )
                setattr(self, f"ubn_{i}_bn_1_scale", nn.Linear(ray_hidden_dim, mid_c))
                setattr(
                    self, f"ubn_{i}_bn_1_translate", nn.Linear(ray_hidden_dim, mid_c)
                )
                setattr(self, f"ubn_{i}_bn_2_scale", nn.Linear(ray_hidden_dim, out_c))
                setattr(
                    self, f"ubn_{i}_bn_2_translate", nn.Linear(ray_hidden_dim, out_c)
                )

        if isinstance(self.n_classes, int):
            self.final_convt_task_0_weights = nn.Linear(
                ray_hidden_dim, 18 * self.n_classes * 3 * 3
            )
            # self.final_convt_task_1_weights = nn.Linear(ray_hidden_dim, 18 * 1 * 3 * 3)
        # else:
        #     self.final_convt_task_0_weights = nn.Linear(
        #         ray_hidden_dim, 16 * self.n_classes[0] * 3 * 3
        #     )
        #     self.final_convt_task_1_weights = nn.Linear(
        #         ray_hidden_dim, 16 * self.n_classes[1] * 3 * 3
        #     )

    def forward(self, ray):

        features = self.ray_mlp(ray)

        out_dict = dict()

        out_dict["init.conv.weight"] = self.initial_block_weights(features).reshape(13, 5, 3, 3),
        out_dict["init.conv.bias"] = self.initial_block_bias(features).flatten(),

        for i in range(self.n_rddn):
            in_c, mid_c, out_c = (
                self.rddn_c[i][0],
                self.rddn_c[i][0] // 4,
                self.rddn_c[i][1],
            )
            out_dict[f"rddn_{i}.conv1.weight"] = getattr(self, f"rddn_{i}_conv_0_weights")(
                features
            ).reshape(mid_c, in_c, 1, 1),
            out_dict[f"rddn_{i}.conv2.weight"] = getattr(self, f"rddn_{i}_conv_1_weights")(
                features
            ).reshape(mid_c, mid_c, 3, 3),
            out_dict[f"rddn_{i}.conv2.bias"] = getattr(self, f"rddn_{i}_conv_1_bias")(
                features
            ).flatten(),
            out_dict[f"rddn_{i}.conv3.weight"] = getattr(self, f"rddn_{i}_conv_2_weights")(
                features
            ).reshape(out_c, mid_c, 1, 1),

            if (self.bn_mode == "hbn") and (i % 2 == 0):
                out_dict["bn_0_scale"] = getattr(self, f"rddn_{i}_bn_0_scale")(
                    features
                ).reshape(mid_c, 1, 1)
                out_dict["bn_0_translate"] = getattr(
                    self, f"rddn_{i}_bn_0_translate"
                )(features).reshape(mid_c, 1, 1)
                out_dict["bn_1_scale"] = getattr(self, f"rddn_{i}_bn_1_scale")(
                    features
                ).reshape(mid_c, 1, 1)
                out_dict["bn_1_translate"] = getattr(
                    self, f"rddn_{i}_bn_1_translate"
                )(features).reshape(mid_c, 1, 1)
                out_dict["bn_2_scale"] = getattr(self, f"rddn_{i}_bn_2_scale")(
                    features
                ).reshape(out_c, 1, 1)
                out_dict["bn_2_translate"] = getattr(
                    self, f"rddn_{i}_bn_2_translate"
                )(features).reshape(out_c, 1, 1)

        for i in range(self.n_asn):
            out_dict[f"asn_{i}.conv1.weight"] = getattr(self, f"asn_{i}_conv_0_weights")(
                features
            ).reshape(36, 144, 1, 1),
            out_dict[f"asn_{i}.conv21.weight"] = getattr(self, f"asn_{i}_conv_11_weights")(
                features
            ).reshape(36, 36, 1, 5),
            out_dict[f"asn_{i}.conv22.weight"] = getattr(self, f"asn_{i}_conv_12_weights")(
                features
            ).reshape(36, 36, 5, 1),
            out_dict[f"asn_{i}.conv3.weight"] = getattr(self, f"asn_{i}_conv_2_weights")(
                features
            ).reshape(144, 36, 1, 1),

            if (self.bn_mode == "hbn") and (i % 2 == 0):
                out_dict["bn_0_scale"] = getattr(self, f"asn_{i}_bn_0_scale")(
                    features
                ).reshape(36, 1, 1)
                out_dict["bn_0_translate"] = getattr(
                    self, f"asn_{i}_bn_0_translate"
                )(features).reshape(36, 1, 1)
                out_dict["bn_1_scale"] = getattr(self, f"asn_{i}_bn_1_scale")(
                    features
                ).reshape(36, 1, 1)
                out_dict["bn_1_translate"] = getattr(
                    self, f"asn_{i}_bn_1_translate"
                )(features).reshape(36, 1, 1)
                out_dict["bn_2_scale"] = getattr(self, f"asn_{i}_bn_2_scale")(
                    features
                ).reshape(144, 1, 1)
                out_dict["bn_2_translate"] = getattr(
                    self, f"asn_{i}_bn_2_translate"
                )(features).reshape(144, 1, 1)

        for i in range(self.n_ubn):
            in_c, mid_c, out_c = (
                self.ubn_c[i][0],
                self.ubn_c[i][0] // 4,
                self.ubn_c[i][1],
            )
            out_dict[f"ubn_{i}.convt1.weight"] = getattr(self, f"ubn_{i}_convt_0_weights")(
                features
            ).reshape(in_c, mid_c, 1, 1),
            out_dict[f"ubn_{i}.convt2.weight"] = getattr(self, f"ubn_{i}_convt_1_weights")(
                features
            ).reshape(mid_c, mid_c, 3, 3),
            out_dict[f"ubn_{i}.convt3.weight"] = getattr(self, f"ubn_{i}_convt_2_weights")(
                features
            ).reshape(mid_c, out_c, 1, 1),
            out_dict[f"ubn_{i}.main_conv.weight"] = getattr(self, f"ubn_{i}_conv_0_weights")(
                features
            ).reshape(out_c, in_c, 1, 1),
            out_dict[f"ubn_{i}.main_conv.bias"] = getattr(self, f"ubn_{i}_conv_0_bias")(
                features
            ).flatten(),

            if (self.bn_mode == "hbn") and (i % 2 == 0):
                out_dict["bn_0_scale"] = getattr(self, f"ubn_{i}_bn_0_scale")(
                    features
                ).reshape(mid_c, 1, 1)
                out_dict["bn_0_translate"] = getattr(
                    self, f"ubn_{i}_bn_0_translate"
                )(features).reshape(mid_c, 1, 1)
                out_dict["bn_1_scale"] = getattr(self, f"ubn_{i}_bn_1_scale")(
                    features
                ).reshape(mid_c, 1, 1)
                out_dict["bn_1_translate"] = getattr(
                    self, f"ubn_{i}_bn_1_translate"
                )(features).reshape(mid_c, 1, 1)
                out_dict["bn_2_scale"] = getattr(self, f"ubn_{i}_bn_2_scale")(
                    features
                ).reshape(out_c, 1, 1)
                out_dict["bn_2_translate"] = getattr(
                    self, f"ubn_{i}_bn_2_translate"
                )(features).reshape(out_c, 1, 1)

        out_dict["fullconv.weight"] = getattr(self, "final_convt_task_0_weights")(
            features
        ).reshape(18, self.n_classes, 3, 3),

        return out_dict

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])

import torch
import torch.nn as nn
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from collections import OrderedDict

import torch
import torch.nn as nn
from collections import OrderedDict

class EnetHyper1(nn.Module):
    def __init__(self, ray_hidden_dim=100, num_rdnn_layers=0):
        super().__init__()
        self.rddn_c=rddn_channels
        self.n_rddn = num_rdnn_layers
        
        self.ray_mlp = nn.Sequential(
            nn.Linear(25, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, ray_hidden_dim),
        )

        # Only include the layers required for the initial block
        self.initial_block_weights = nn.Linear(ray_hidden_dim, 25 * 13 * 3 * 3)
        self.initial_block_bias = nn.Linear(ray_hidden_dim, 13)
        self.prelu_weight = nn.Linear(ray_hidden_dim, 38)
        self.conv_1x1_weight = nn.Linear(ray_hidden_dim, 18 * 38)  
        self.conv_1x1_bias = nn.Linear(ray_hidden_dim, 18)
        self.initial_batchnorm_weight = nn.Linear(ray_hidden_dim, 13)
        self.initial_batchnorm_bias = nn.Linear(ray_hidden_dim, 13)
        self.initial_batchnorm_running_mean = nn.Linear(ray_hidden_dim, 13)
        self.initial_batchnorm_running_var = nn.Linear(ray_hidden_dim, 13)
        
        for i in range(self.n_rddn):
            in_c, mid_c, out_c = (
                self.rddn_c[i][0],
                self.rddn_c[i][0] // 4,
                self.rddn_c[i][1],
            )
            setattr(
                self,
                f"rddn_{i}_conv_0_weights",
                nn.Linear(ray_hidden_dim, mid_c * in_c * 1 * 1),
            )
            setattr(
                self,
                f"rddn_{i}_conv_1_weights",
                nn.Linear(ray_hidden_dim, mid_c * mid_c * 3 * 3),
            )
            setattr(self, f"rddn_{i}_conv_1_bias", nn.Linear(ray_hidden_dim, mid_c))
            setattr(
                self,
                f"rddn_{i}_conv_2_weights",
                nn.Linear(ray_hidden_dim, out_c * mid_c * 1 * 1),
            )
            setattr(self, f"rddn_{i}_prelu1_weight", nn.Linear(ray_hidden_dim, 1))
            #seperate prelu2 and prelu3 not needed as all 3 prelus are same
            # setattr(self, f"rddn_{i}_prelu2_weight", nn.Linear(ray_hidden_dim, 1))
            # setattr(self, f"rddn_{i}_prelu3_weight", nn.Linear(ray_hidden_dim, 1))
            setattr(self, f"rddn_{i}_bn_0_weights", nn.Linear(ray_hidden_dim, mid_c))
            setattr(self, f"rddn_{i}_bn_0_bias", nn.Linear(ray_hidden_dim, mid_c))
            setattr(self, f"rddn_{i}_bn_0_running_mean", nn.Linear(ray_hidden_dim, mid_c))
            setattr(self, f"rddn_{i}_bn_0_running_var", nn.Linear(ray_hidden_dim, mid_c))
            
            setattr(self, f"rddn_{i}_bn_1_weights", nn.Linear(ray_hidden_dim, mid_c))
            setattr(self, f"rddn_{i}_bn_1_bias", nn.Linear(ray_hidden_dim, mid_c))
            setattr(self, f"rddn_{i}_bn_1_running_mean", nn.Linear(ray_hidden_dim, mid_c))
            setattr(self, f"rddn_{i}_bn_1_running_var", nn.Linear(ray_hidden_dim, mid_c))
            
            setattr(self, f"rddn_{i}_bn_2_weights", nn.Linear(ray_hidden_dim, out_c))
            setattr(self, f"rddn_{i}_bn_2_bias", nn.Linear(ray_hidden_dim, out_c))
            setattr(self, f"rddn_{i}_bn_2_running_mean", nn.Linear(ray_hidden_dim, out_c))
            setattr(self, f"rddn_{i}_bn_2_running_var", nn.Linear(ray_hidden_dim, out_c))
            
    def forward(self, ray):
        features = self.ray_mlp(ray)
        w1 = self.initial_block_weights(features).reshape(13, 25, 3, 3)
        
        one_hot_indices = torch.nonzero(ray[0]).flatten()
        
        w1_selected = w1[:, one_hot_indices, :, :]

        prelu_weights = self.prelu_weight(features).flatten()
        
        selected_prelu_weights = torch.cat((prelu_weights[one_hot_indices], prelu_weights[-13:]))

        # Indices for the selected channels (from one-hot indices and last 13)
        selected_indices = torch.cat((one_hot_indices, torch.arange(len(prelu_weights) - 13, len(prelu_weights), device=ray.device)))
        # print(selected_indices)
        conv_1x1_weights = self.conv_1x1_weight(features).reshape(18, 38, 1, 1)[:, selected_indices, :, :]
        conv_1x1_bias = self.conv_1x1_bias(features).flatten()
        
        batchnorm_weight = self.initial_batchnorm_weight(features).flatten()
        batchnorm_bias = self.initial_batchnorm_bias(features).flatten()
        batchnorm_running_mean = self.initial_batchnorm_running_mean(features).flatten()
        batchnorm_running_var = self.initial_batchnorm_running_var(features).flatten()

        out_dict = {
            "init.conv.weight": w1_selected,
            "init.conv.bias": self.initial_block_bias(features).flatten(),
            # "init.prelu.weight": selected_prelu_weights, 
            "init.conv_1x1.weight": conv_1x1_weights,
            "init.conv_1x1.bias": conv_1x1_bias,
            "init.batchnorm.weight": batchnorm_weight,
            "init.batchnorm.bias": batchnorm_bias,
            # "init.prelu.weight": selected_prelu_weights,
            # "init.batchnorm.running_mean": batchnorm_running_mean,
            # "init.batchnorm.running_var": batchnorm_running_var,
        }
        
        for i in range(self.n_rddn):
            in_c, mid_c, out_c = (
                self.rddn_c[i][0],
                self.rddn_c[i][0] // 4,
                self.rddn_c[i][1],
            )
            out_dict[f"rddn_{i}.conv1.weight"] = getattr(self, f"rddn_{i}_conv_0_weights")(features).reshape(mid_c, in_c, 1, 1)
            out_dict[f"rddn_{i}.conv2.weight"] = getattr(self, f"rddn_{i}_conv_1_weights")(features).reshape(mid_c, mid_c, 3, 3)
            out_dict[f"rddn_{i}.conv2.bias"] = getattr(self, f"rddn_{i}_conv_1_bias")(features).flatten()
            out_dict[f"rddn_{i}.conv3.weight"] = getattr(self, f"rddn_{i}_conv_2_weights")(features).reshape(out_c, mid_c, 1, 1)
            out_dict[f"rddn_{i}.batchnorm1.weight"] = getattr(self, f"rddn_{i}_bn_0_weights")(features).flatten()
            out_dict[f"rddn_{i}.batchnorm1.bias"] = getattr(self, f"rddn_{i}_bn_0_bias")(features).flatten()
            # out_dict[f"rddn_{i}.batchnorm1.running_mean"] = getattr(self, f"rddn_{i}_bn_0_running_mean")(features).flatten()
            # out_dict[f"rddn_{i}.batchnorm1.running_var"] = getattr(self, f"rddn_{i}_bn_0_running_var")(features).flatten()
            out_dict[f"rddn_{i}.batchnorm2.weight"] = getattr(self, f"rddn_{i}_bn_1_weights")(features).flatten()
            out_dict[f"rddn_{i}.batchnorm2.bias"] = getattr(self, f"rddn_{i}_bn_1_bias")(features).flatten()
            # out_dict[f"rddn_{i}.batchnorm2.running_mean"] = getattr(self, f"rddn_{i}_bn_1_running_mean")(features).flatten()
            # out_dict[f"rddn_{i}.batchnorm2.running_var"] = getattr(self, f"rddn_{i}_bn_1_running_var")(features).flatten()
            out_dict[f"rddn_{i}.batchnorm3.weight"] = getattr(self, f"rddn_{i}_bn_2_weights")(features).flatten()
            out_dict[f"rddn_{i}.batchnorm3.bias"] = getattr(self, f"rddn_{i}_bn_2_bias")(features).flatten()
            # out_dict[f"rddn_{i}.batchnorm3.running_mean"] = getattr(self, f"rddn_{i}_bn_2_running_mean")(features).flatten()
            # out_dict[f"rddn_{i}.batchnorm3.running_var"] = getattr(self, f"rddn_{i}_bn_2_running_var")(features).flatten()
            # out_dict[f"rddn_{i}.prelu1.weight"] = getattr(self, f"rddn_{i}_prelu1_weight")(features).flatten()
            # out_dict[f"rddn_{i}.prelu2.weight"] = getattr(self, f"rddn_{i}_prelu1_weight")(features).flatten()
            # out_dict[f"rddn_{i}.prelu3.weight"] = getattr(self, f"rddn_{i}_prelu1_weight")(features).flatten()

        return out_dict

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])


        return out_dict

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])


    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])


    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])

class ENetAda(nn.Module):
    def __init__(self, n_classes, device):
        super().__init__()

        # Define class variables
        self.C = n_classes
        self.device = device

        # The initial block
        self.init = InitialBlock()

        # The first bottleneck
        self.b10 = RDDNeck(
            dilation=1,
            in_channels=18,
            out_channels=72,
            down_flag=True,
            p=0.01,
            device=self.device,
        )

        self.b11 = RDDNeck(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
        )

        self.b12 = RDDNeck(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
        )

        self.b13 = RDDNeck(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
        )

        self.b14 = RDDNeck(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
            ray_dim=2,
        )

        # The second bottleneck
        self.b20 = RDDNeck(
            dilation=1,
            in_channels=72,
            out_channels=144,
            down_flag=True,
            device=self.device,
        )

        self.b21 = RDDNeck(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b22 = RDDNeck(
            dilation=2,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b23 = ASNeck(in_channels=144, out_channels=144)

        self.b24 = RDDNeck(
            dilation=4,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b25 = RDDNeck(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b26 = RDDNeck(
            dilation=8,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b27 = ASNeck(in_channels=144, out_channels=144)

        self.b28 = RDDNeck(
            dilation=16,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            ray_dim=2,
        )

        # The third bottleneck
        self.b31 = RDDNeck(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b32 = RDDNeck(
            dilation=2,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b33 = ASNeck(in_channels=144, out_channels=144)

        self.b34 = RDDNeck(
            dilation=4,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b35 = RDDNeck(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b36 = RDDNeck(
            dilation=8,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.b37 = ASNeck(in_channels=144, out_channels=144)

        self.b38 = RDDNeck(
            dilation=16,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
            ray_dim=2,
        )

        # The fourth bottleneck
        self.b40 = UBNeck(in_channels=144, out_channels=72, relu=True)

        self.b41 = RDDNeck(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            relu=True,
            device=self.device,
        )

        self.b42 = RDDNeck(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            relu=True,
            device=self.device,
        )

        # The fifth bottleneck
        self.b50 = UBNeck(in_channels=72, out_channels=18, relu=True)

        self.b51 = RDDNeck(
            dilation=1,
            in_channels=18,
            out_channels=18,
            down_flag=False,
            relu=True,
            device=self.device,
        )

        # Final ConvTranspose Layer
        self.fullconv = nn.ConvTranspose2d(
            in_channels=18,
            out_channels=self.C,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

    def forward(self, x, ray=None):
        # The initial block
        x = self.init(x)

        # The first bottleneck
        x, i1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x, ray)

        # The second bottleneck
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x, ray)

        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x, ray)

        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)

        # Final ConvTranspose Layer
        out = self.fullconv1(x)
        return out


class ENetVanila(nn.Module):
    def __init__(self, num_classes, device, in_channels=3):
        super().__init__()

        # Define class variables
        self.C = num_classes
        self.device = device
        self.in_channels = in_channels

        # The initial block
        self.init = InitialBlockVanila(in_channels=self.in_channels)

        # The first bottleneck
        self.rddn_0 = RDDNeckVanila(
            dilation=1,
            in_channels=18,
            out_channels=72,
            down_flag=True,
            p=0.01,
            device=self.device,
        )

        self.rddn_1 = RDDNeckVanila(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
        )

        self.rddn_2 = RDDNeckVanila(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
        )

        self.rddn_3 = RDDNeckVanila(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
        )

        self.rddn_4 = RDDNeckVanila(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            p=0.01,
            device=self.device,
        )

        # The second bottleneck
        self.rddn_5 = RDDNeckVanila(
            dilation=1,
            in_channels=72,
            out_channels=144,
            down_flag=True,
            device=self.device,
        )

        self.rddn_6 = RDDNeckVanila(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.rddn_7 = RDDNeckVanila(
            dilation=2,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.asn_0 = ASNeckVanila(in_channels=144, out_channels=144)

        self.rddn_8 = RDDNeckVanila(
            dilation=4,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.rddn_9 = RDDNeckVanila(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.rddn_10 = RDDNeckVanila(
            dilation=8,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.asn_1 = ASNeckVanila(in_channels=144, out_channels=144)

        self.rddn_11 = RDDNeckVanila(
            dilation=16,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        # The third bottleneck
        self.rddn_12 = RDDNeckVanila(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.rddn_13 = RDDNeckVanila(
            dilation=2,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.asn_2 = ASNeckVanila(in_channels=144, out_channels=144)

        self.rddn_14 = RDDNeckVanila(
            dilation=4,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.rddn_15 = RDDNeckVanila(
            dilation=1,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.rddn_16 = RDDNeckVanila(
            dilation=8,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        self.asn_3 = ASNeckVanila(in_channels=144, out_channels=144)

        self.rddn_17 = RDDNeckVanila(
            dilation=16,
            in_channels=144,
            out_channels=144,
            down_flag=False,
            device=self.device,
        )

        # The fourth bottleneck
        self.ubn_0 = UBNeckVanila(in_channels=144, out_channels=72, relu=True)

        self.rddn_18 = RDDNeckVanila(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            relu=True,
            device=self.device,
        )

        self.rddn_19 = RDDNeckVanila(
            dilation=1,
            in_channels=72,
            out_channels=72,
            down_flag=False,
            relu=True,
            device=self.device,
        )

        # The fifth bottleneck
        self.ubn_1 = UBNeckVanila(in_channels=72, out_channels=18, relu=True)

        self.rddn_20 = RDDNeckVanila(
            dilation=1,
            in_channels=18,
            out_channels=18,
            down_flag=False,
            relu=True,
            device=self.device,
        )

        # Final ConvTranspose Layer
        self.fullconv = nn.ConvTranspose2d(
            in_channels=18,
            out_channels=self.C,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

        initialize_weights(self)

    def forward(self, x):
        # The initial block
        #print("Input to INITBLOCK in EnET Vanila ", x.shape)
        x = self.init(x)

        # The first bottleneck
        x, i1 = self.rddn_0(x)
        x = self.rddn_1(x)
        x = self.rddn_2(x)
        x = self.rddn_3(x)
        x = self.rddn_4(x)

        # The second bottleneck
        x, i2 = self.rddn_5(x)
        x = self.rddn_6(x)
        x = self.rddn_7(x)
        x = self.asn_0(x)
        x = self.rddn_8(x)
        x = self.rddn_9(x)
        x = self.rddn_10(x)
        x = self.asn_1(x)
        x = self.rddn_11(x)

        # The third bottleneck
        x = self.rddn_12(x)
        x = self.rddn_13(x)
        x = self.asn_2(x)
        x = self.rddn_14(x)
        x = self.rddn_15(x)
        x = self.rddn_16(x)
        x = self.asn_3(x)
        x = self.rddn_17(x)

        # The fourth bottleneck
        x = self.ubn_0(x, i2)
        x = self.rddn_18(x)
        x = self.rddn_19(x)

        # The fifth bottleneck
        x = self.ubn_1(x, i1)
        x = self.rddn_20(x)

        # Final ConvTranspose Layer
        # if len(self.C) > 1:
        #     t1_out = self.fullconv1(x)
        #     t2_out = self.fullconv2(x)
        #     return t1_out, t2_out
        # else:
        #     out = self.fullconv(x)
        #     return out
        out = self.fullconv(x)
        return out


def validate_weights_sd(net, weights):
    set_diff = set(net.state_dict().keys()) - set(weights.keys())
    missing_layers = []
    for i in set_diff:
        if "batchnorm" in i:
            continue
        elif "prelu" in i:
            continue
        else:
            missing_layers.append(i)
    print(missing_layers)


def load_weights_sd(net, weights):
    for layer_name, layer_params in net.named_parameters():
        if layer_name in weights:
            layer_params.data.copy_(weights[layer_name])
        else:
            # print(f"Skipping layer {layer_name} because it does not exist in the provided weights.")
            continue

if __name__ == "__main__":
    device = torch.device("cpu")
    hnet = EnetHyper(n_classes=12).to(device)
    net = ENetVanila(num_classes=12, device=device).to(device)
    ray = torch.tensor([0.5, 0.5]).to(device)
    input = torch.rand(1, 5, 216, 408).to(device)
    num_classes = 12
    label = torch.randint(0, 4, (1, 216, 408)).to(torch.device("cpu"))
    weights = hnet(ray)
    weights = OrderedDict({key: value[0] for key, value in weights.items()})
    validate_weights_sd(net, weights)
    load_weights_sd(net, weights)
    output = net(input)
    print("debug")
