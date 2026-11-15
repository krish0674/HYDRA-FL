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
