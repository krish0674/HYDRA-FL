"""Model factory.

Two backbones are supported, each with a matching hypernetwork:

* ``enet``: ``ENetVanila`` client/global model and ``EnetHyper1`` hypernetwork
  (generates the initial block and the first ``num_hn_layers`` RDDN bottlenecks).
* ``unet``: ``segmentation_models_pytorch.Unet`` with a ResNet-18 encoder and the
  ``UnetHyper`` hypernetwork (generates the encoder stem and the first
  ``num_hn_layers`` encoder stages). ``segmentation_models_pytorch`` is imported
  lazily so the ENet path has no dependency on it.
"""

from ..models.ENet import ENetVanila, EnetHyper1
from ..models.UNetHyper import UnetHyper

BACKBONES = ("enet", "unet")


def get_model(name, num_classes, device="cpu", in_channels=3, num_rdnn_layers=0, pretrained=True):
    """Original entry point, kept for backwards compatibility.

    ``name`` is one of ``ENetVanila``, ``EnetHyper``, ``Unet``, ``UN``.
    """
    if name == "ENetVanila":
        return ENetVanila(num_classes, device=device, in_channels=in_channels).to(device)
    if name == "EnetHyper":
        return EnetHyper1(num_rdnn_layers=num_rdnn_layers).to(device)
    if name == "UN":
        return UnetHyper(num_unet_layers=num_rdnn_layers).to(device)
    if name == "Unet":
        from segmentation_models_pytorch import Unet  # lazy import

        return Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            activation=None,
            in_channels=in_channels,
        ).to(device)
    raise ValueError("Model {} not available".format(name))


def get_target_model(backbone, num_classes, device, in_channels, pretrained=True):
    """Client / global segmentation model for ``backbone`` in {'enet', 'unet'}."""
    if backbone == "enet":
        return get_model("ENetVanila", num_classes, device=device, in_channels=in_channels)
    if backbone == "unet":
        return get_model("Unet", num_classes, device=device, in_channels=in_channels, pretrained=pretrained)
    raise ValueError("backbone must be one of {}".format(BACKBONES))


def get_hypernet(backbone, device, num_hn_layers=0):
    """Hypernetwork matching ``backbone``."""
    if backbone == "enet":
        return get_model("EnetHyper", None, device=device, num_rdnn_layers=num_hn_layers)
    if backbone == "unet":
        return get_model("UN", None, device=device, num_rdnn_layers=num_hn_layers)
    raise ValueError("backbone must be one of {}".format(BACKBONES))
