import json

import torch

from .data.node import get_data
from .models.model import get_target_model

NUM_BANDS = 25


def read_band_config(config_file, num_clients, all_bands=False):
    """Return a list with one band-index list per client.

    ``config_file`` is a JSON file with keys ``client_0`` ... ``client_{N-1}`` whose
    values are comma-separated band indices (0-24). With ``all_bands=True`` every
    client gets all 25 bands (vanilla FL baseline) and the file is not read.
    """
    if all_bands:
        return [list(range(NUM_BANDS)) for _ in range(num_clients)]
    with open(config_file) as f:
        configuration = json.load(f)
    channels = []
    for c in range(num_clients):
        key = f"client_{c}"
        if key not in configuration:
            raise KeyError(f"{config_file} has no entry '{key}' (num_clients={num_clients})")
        channels.append([int(i) for i in str(configuration[key]).split(",")])
    return channels


def band_mask(channels):
    """Binary band-presence vector b_i (shape 1 x 25) for a list of band indices."""
    mask = torch.zeros(1, NUM_BANDS)
    for k in channels:
        mask[0][int(k)] = 1
    return mask


def prepare_dataset_models(config):
    """Build one dict per client: model, loaders and band mask.

    Expected keys in ``config``: num_clients, device, num_classes, config_file,
    dir, data_folder, label_folder, backbone ('enet' | 'unet'), and optionally
    all_bands, csv_file, batch_size, num_workers, pretrained.
    """
    num_clients = config["num_clients"]
    device = config["device"]
    backbone = config.get("backbone", "enet")

    channels = read_band_config(config["config_file"], num_clients, all_bands=config.get("all_bands", False))

    train_l, val_l, test_l = get_data(
        num_clients,
        config["dir"],
        config["data_folder"],
        config["label_folder"],
        channels,
        csv_file=config.get("csv_file"),
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
        backbone=backbone,
    )

    client_dicts = {}
    for c in range(num_clients):
        client = {}
        client["channels"] = channels[c]
        client["model"] = get_target_model(
            backbone,
            config["num_classes"],
            device=device,
            in_channels=len(channels[c]),
            pretrained=config.get("pretrained", True),
        )
        client["train_loader"], client["val_loader"], client["test_loader"] = train_l[c], val_l[c], test_l[c]
        client["in_channels"] = band_mask(channels[c])
        client["prev_grads"] = None
        client_dicts[c] = client
    return client_dicts
