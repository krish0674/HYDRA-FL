"""Hypernetwork that generates the band-dependent layers of a UNet (ResNet-18 encoder)
client model. Recovered unchanged from branch ``no_prelu_UNet`` of the development
repository (commit e05d524, 4 Sep 2024); it was removed from ``no_prelu`` after the UNet runs.
"""
import torch
import torch.nn as nn

class UnetHyper(nn.Module):
    def __init__(self, ray_hidden_dim=100, num_unet_layers=0):
        super().__init__()

        self.num_unet_layers = num_unet_layers
        
        self.ray_mlp = nn.Sequential(
            nn.Linear(25, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, ray_hidden_dim),
        )

        # Initial block (conv1 and bn1 in the encoder)
        self.init_conv1_weight = nn.Linear(ray_hidden_dim, 64 * 25 * 7 * 7)
        # self.init_conv1_bias = nn.Linear(ray_hidden_dim, 64)
        self.init_bn1_weight = nn.Linear(ray_hidden_dim, 64)
        self.init_bn1_bias = nn.Linear(ray_hidden_dim, 64)

        # Layer 1
        if self.num_unet_layers>0:
            self.layer1_0_bn1_weight = nn.Linear(ray_hidden_dim, 64)
            self.layer1_0_bn1_bias = nn.Linear(ray_hidden_dim, 64)
            self.layer1_0_conv2_weight = nn.Linear(ray_hidden_dim, 64 * 64 * 3 * 3)
            self.layer1_0_bn2_weight = nn.Linear(ray_hidden_dim, 64)
            self.layer1_0_bn2_bias = nn.Linear(ray_hidden_dim, 64)

            self.layer1_1_conv1_weight = nn.Linear(ray_hidden_dim, 64 * 64 * 3 * 3)
            self.layer1_1_bn1_weight = nn.Linear(ray_hidden_dim, 64)
            self.layer1_1_bn1_bias = nn.Linear(ray_hidden_dim, 64)
            self.layer1_1_conv2_weight = nn.Linear(ray_hidden_dim, 64 * 64 * 3 * 3)
            self.layer1_1_bn2_weight = nn.Linear(ray_hidden_dim, 64)
            self.layer1_1_bn2_bias = nn.Linear(ray_hidden_dim, 64)

        # Layers 2, 3, 4
        self.layers_info = {
            2: [64, 128],
            3: [128, 256],
            4: [256, 512]
        }
        
        for i in range(2, 5):
            if self.num_unet_layers >= i:
                in_channels, out_channels = self.layers_info[i]
                
                setattr(self, f"layer{i}_0_conv1_weight", nn.Linear(ray_hidden_dim, out_channels * in_channels * 3 * 3))
                setattr(self, f"layer{i}_0_bn1_weight", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_0_bn1_bias", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_0_conv2_weight", nn.Linear(ray_hidden_dim, out_channels * out_channels * 3 * 3))
                setattr(self, f"layer{i}_0_bn2_weight", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_0_bn2_bias", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_0_downsample_0_weight", nn.Linear(ray_hidden_dim, out_channels * in_channels * 1 * 1))
                setattr(self, f"layer{i}_0_downsample_1_weight", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_0_downsample_1_bias", nn.Linear(ray_hidden_dim, out_channels))

                setattr(self, f"layer{i}_1_conv1_weight", nn.Linear(ray_hidden_dim, out_channels * out_channels * 3 * 3))
                setattr(self, f"layer{i}_1_bn1_weight", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_1_bn1_bias", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_1_conv2_weight", nn.Linear(ray_hidden_dim, out_channels * out_channels * 3 * 3))
                setattr(self, f"layer{i}_1_bn2_weight", nn.Linear(ray_hidden_dim, out_channels))
                setattr(self, f"layer{i}_1_bn2_bias", nn.Linear(ray_hidden_dim, out_channels))

    def expected_keys(self):
        keys = []

        # Initial block keys
        keys.append('encoder.conv1.weight')
        keys.append('encoder.conv1.bias')
        keys.append('encoder.bn1.weight')
        keys.append('encoder.bn1.bias')

        # Layer 1 keys
        if self.num_unet_layers >= 1:
            keys += [
                'encoder.layer1.0.bn1.weight',
                'encoder.layer1.0.bn1.bias',
                'encoder.layer1.0.conv2.weight',
                'encoder.layer1.0.bn2.weight',
                'encoder.layer1.0.bn2.bias',
                'encoder.layer1.1.conv1.weight',
                'encoder.layer1.1.bn1.weight',
                'encoder.layer1.1.bn1.bias',
                'encoder.layer1.1.conv2.weight',
                'encoder.layer1.1.bn2.weight',
                'encoder.layer1.1.bn2.bias',
            ]

        # Layers 2, 3, 4 keys
        for i in range(2, 5):
            if self.num_unet_layers >= i:
                keys += [
                    f'encoder.layer{i}.0.conv1.weight',
                    f'encoder.layer{i}.0.bn1.weight',
                    f'encoder.layer{i}.0.bn1.bias',
                    f'encoder.layer{i}.0.conv2.weight',
                    f'encoder.layer{i}.0.bn2.weight',
                    f'encoder.layer{i}.0.bn2.bias',
                    f'encoder.layer{i}.0.downsample.0.weight',
                    f'encoder.layer{i}.0.downsample.1.weight',
                    f'encoder.layer{i}.0.downsample.1.bias',
                    f'encoder.layer{i}.1.conv1.weight',
                    f'encoder.layer{i}.1.bn1.weight',
                    f'encoder.layer{i}.1.bn1.bias',
                    f'encoder.layer{i}.1.conv2.weight',
                    f'encoder.layer{i}.1.bn2.weight',
                    f'encoder.layer{i}.1.bn2.bias',
                ]
        
        return keys

    def forward(self, ray):
        features = self.ray_mlp(ray)
        w1 = self.init_conv1_weight(features).reshape(64, 25, 7, 7)
        
        one_hot_indices = torch.nonzero(ray[0]).flatten()
        
        w1_selected = w1[:, one_hot_indices, :, :]
        # Initial block weights and biases
        out_dict = {
            'encoder.conv1.weight': w1_selected,
            # 'encoder.conv1.bias': self.init_conv1_bias(features).flatten(),
            'encoder.bn1.weight': self.init_bn1_weight(features).flatten(),
            'encoder.bn1.bias': self.init_bn1_bias(features).flatten(),
        }

        # Layer 1
        if self.num_unet_layers >= 1:
            out_dict.update({
                'encoder.layer1.0.bn1.weight': self.layer1_0_bn1_weight(features).flatten(),
                'encoder.layer1.0.bn1.bias': self.layer1_0_bn1_bias(features).flatten(),
                'encoder.layer1.0.conv2.weight': self.layer1_0_conv2_weight(features).reshape(64, 64, 3, 3),
                'encoder.layer1.0.bn2.weight': self.layer1_0_bn2_weight(features).flatten(),
                'encoder.layer1.0.bn2.bias': self.layer1_0_bn2_bias(features).flatten(),
                'encoder.layer1.1.conv1.weight': self.layer1_1_conv1_weight(features).reshape(64, 64, 3, 3),
                'encoder.layer1.1.bn1.weight': self.layer1_1_bn1_weight(features).flatten(),
                'encoder.layer1.1.bn1.bias': self.layer1_1_bn1_bias(features).flatten(),
                'encoder.layer1.1.conv2.weight': self.layer1_1_conv2_weight(features).reshape(64, 64, 3, 3),
                'encoder.layer1.1.bn2.weight': self.layer1_1_bn2_weight(features).flatten(),
                'encoder.layer1.1.bn2.bias': self.layer1_1_bn2_bias(features).flatten(),
            })

        # Layers 2, 3, 4
        for i in range(2, 5):
            if self.num_unet_layers >= i:
                out_dict.update({
                    f'encoder.layer{i}.0.conv1.weight': getattr(self, f"layer{i}_0_conv1_weight")(features).reshape(self.layers_info[i][1], self.layers_info[i][0], 3, 3),
                    f'encoder.layer{i}.0.bn1.weight': getattr(self, f"layer{i}_0_bn1_weight")(features).flatten(),
                    f'encoder.layer{i}.0.bn1.bias': getattr(self, f"layer{i}_0_bn1_bias")(features).flatten(),
                    f'encoder.layer{i}.0.conv2.weight': getattr(self, f"layer{i}_0_conv2_weight")(features).reshape(self.layers_info[i][1], self.layers_info[i][1], 3, 3),
                    f'encoder.layer{i}.0.bn2.weight': getattr(self, f"layer{i}_0_bn2_weight")(features).flatten(),
                    f'encoder.layer{i}.0.bn2.bias': getattr(self, f"layer{i}_0_bn2_bias")(features).flatten(),
                    f'encoder.layer{i}.0.downsample.0.weight': getattr(self, f"layer{i}_0_downsample_0_weight")(features).reshape(self.layers_info[i][1], self.layers_info[i][0], 1, 1),
                    f'encoder.layer{i}.0.downsample.1.weight': getattr(self, f"layer{i}_0_downsample_1_weight")(features).flatten(),
                    f'encoder.layer{i}.0.downsample.1.bias': getattr(self, f"layer{i}_0_downsample_1_bias")(features).flatten(),
                    
                    f'encoder.layer{i}.1.conv1.weight': getattr(self, f"layer{i}_1_conv1_weight")(features).reshape(self.layers_info[i][1], self.layers_info[i][1], 3, 3),
                    f'encoder.layer{i}.1.bn1.weight': getattr(self, f"layer{i}_1_bn1_weight")(features).flatten(),
                    f'encoder.layer{i}.1.bn1.bias': getattr(self, f"layer{i}_1_bn1_bias")(features).flatten(),
                    f'encoder.layer{i}.1.conv2.weight': getattr(self, f"layer{i}_1_conv2_weight")(features).reshape(self.layers_info[i][1], self.layers_info[i][1], 3, 3),
                    f'encoder.layer{i}.1.bn2.weight': getattr(self, f"layer{i}_1_bn2_weight")(features).flatten(),
                    f'encoder.layer{i}.1.bn2.bias': getattr(self, f"layer{i}_1_bn2_bias")(features).flatten(),
                })

        return out_dict


