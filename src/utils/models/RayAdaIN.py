import torch.nn as nn


class RayAdaIN(nn.Module):
    def __init__(self, ray_dim, hidden_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.IN = nn.InstanceNorm2d(num_features=num_features, affine=False)
        self.scale = nn.Sequential(
            nn.Linear(ray_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )
        self.translate = nn.Sequential(
            nn.Linear(ray_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )

    def forward(self, input, ray):
        in_out = self.IN(input)
        bs = in_out.shape[0]
        ray_scale = self.scale(ray)
        ray_scale = ray_scale.repeat(bs, 1).reshape(bs, self.num_features, 1, 1)

        ray_translate = self.translate(ray)
        ray_translate = ray_translate.repeat(bs, 1).reshape(bs, self.num_features, 1, 1)

        return in_out * ray_scale + ray_translate


class RayAdaBN(nn.Module):
    def __init__(self, ray_dim, hidden_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features=num_features, affine=True)
        # self.scale = nn.Linear(ray_dim, hidden_dim)
        self.scale = nn.Sequential(
            nn.Linear(ray_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )
        self.translate = nn.Sequential(
            nn.Linear(ray_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )

    def forward(self, input, ray):
        bn_out = self.bn(input)
        bs = bn_out.shape[0]
        ray_scale = self.scale(ray)
        ray_scale = ray_scale.repeat(bs, 1).reshape(bs, self.num_features, 1, 1)
        ray_translate = self.translate(ray)
        ray_translate = ray_translate.repeat(bs, 1).reshape(bs, self.num_features, 1, 1)
        return ray_scale * bn_out + ray_translate


class HyperRayAdaBN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features=num_features, affine=False)

    def forward(self, input, scale, translate):
        bn_out = self.bn(input)
        return scale * bn_out + translate
