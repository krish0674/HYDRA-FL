"""Helpers shared by train.py and infer.py.

Everything here mirrors the behaviour of the original HYDRA-FL runs
(branch ``no_prelu`` of the 2024 development repository); only device
handling and logging were generalised.
"""

import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    """Seed Python, NumPy and PyTorch. ``None`` leaves everything unseeded."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_weights_sd(net, weights):
    """Copy the tensors in ``weights`` into the parameters / buffers of ``net``
    whose names match. Tensors are copied with ``.data.copy_`` so no autograd
    graph is attached to the client model (the stop-gradient of Algorithm 1)."""
    for layer_name, layer_params in net.named_parameters():
        if layer_name in weights:
            layer_params.data.copy_(weights[layer_name])
    for layer_name, layer_params in net.named_buffers():
        if layer_name in weights:
            layer_params.data.copy_(weights[layer_name])


def replace_batchnorm(device, module: nn.Module):
    """Replace every BatchNorm2d with an affine InstanceNorm2d (used in all
    reported runs, see the paper, Section V-A)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.InstanceNorm2d(child.num_features, affine=True).to(device))
        else:
            replace_batchnorm(device, child)


def replace_prelu(device, module: nn.Module):
    """Replace every PReLU with ReLU (used in all reported runs)."""
    for name, child in module.named_children():
        if isinstance(child, nn.PReLU):
            setattr(module, name, nn.ReLU().to(device))
        else:
            replace_prelu(device, child)


def hypernet_loss(generated, trained):
    """Loss used to train the hypernetwork (line 8 of Algorithm 1).

    ``generated[i]`` holds the tensors produced by the hypernetwork for client i
    (with autograd graph), ``trained[i]`` the same tensors after local training
    (detached). The loss is the sum of squared differences over all entries and
    all clients; each pair of tensors has identical shape by construction, so
    clients with different band counts are never compared with each other.
    Note: this is a plain sum, not divided by the number of clients or entries,
    exactly as in the original runs (the scale is absorbed by the learning rate).
    """
    loss = 0.0
    for gen, tr in zip(generated, trained):
        for key in gen.keys():
            loss = loss + torch.sum((gen[key] - tr[key].detach()) ** 2)
    return loss


def make_dice(num_classes, device):
    """Micro-averaged multiclass Dice over all classes (background included), as
    used for every number reported in the paper.

    The original runs used ``torchmetrics.classification.Dice(average='micro',
    multiclass=True, num_classes=11)`` (torchmetrics 1.3.2). That class was removed
    in torchmetrics 1.8; ``MulticlassF1Score(average='micro')`` computes the identical
    quantity (micro Dice == micro F1), so it is used as the fallback. Note that the
    micro average over *all* classes of a single-label segmentation equals the
    overall pixel accuracy.
    """
    try:
        from torchmetrics.classification import Dice  # torchmetrics < 1.8

        return Dice(average="micro", multiclass=True, num_classes=num_classes).to(device)
    except ImportError:
        from torchmetrics.classification import MulticlassF1Score

        return MulticlassF1Score(num_classes=num_classes, average="micro").to(device)


def dice_from_logits(logits, target, metric):
    pred = torch.argmax(logits, dim=1)
    return metric(pred, target)


@torch.no_grad()
def evaluate(model, loader, metric, device):
    """Mean of the per-batch Dice over ``loader`` (batch size 1 for the common
    validation / test sets, so this is a per-image average)."""
    model.eval()
    scores = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).long()
        scores.append(dice_from_logits(model(images), labels, metric).item())
    return float(np.mean(scores)) if scores else float("nan")


def hypernet_keys(hypernet, device, num_bands=25):
    """Names of the state-dict entries the hypernetwork generates (found by one
    forward pass with the all-bands mask)."""
    with torch.no_grad():
        out = hypernet(torch.ones((1, num_bands), dtype=torch.float32, device=device))
    return list(OrderedDict(out).keys())
