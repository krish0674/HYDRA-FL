"""Federated training loop for HYDRA-FL and for the vanilla FL baselines.

One server round (Algorithm 1 of the paper):

1. For every client i, the hypernetwork H_phi receives the client's band mask b_i
   and generates the weights of the band-dependent initial layers; these are loaded
   into the client model together with the current global shared layers.
2. The client trains locally for E epochs (FedAvg: cross-entropy with Adam;
   FedProx: cross-entropy + (mu/2)||w - w_global||^2 with SGD-momentum) and
   returns its weights.
3. The server updates H_phi with the squared error between the generated initial
   layers and the locally trained ones (clients' weights are constants).
4. The shared layers are averaged into the global model.
5. The global model used for evaluation takes its initial layers from H_phi
   applied to the all-bands mask and is scored on the common val / test splits.

With ``use_hypernet=False`` steps 1 and 3 are skipped and all layers are averaged;
this is the vanilla FedAvg / FedProx baseline of the paper, which requires every
client to hold the same bands (use ``all_bands=True`` for the 25-band baseline).
"""

import csv
import importlib
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.datasetinit import get_datasets, split_bands
from .hn_utils import (
    evaluate,
    hypernet_keys,
    hypernet_loss,
    load_weights_sd,
    make_dice,
    replace_batchnorm,
    replace_prelu,
    set_seed,
)
from .models.model import get_hypernet, get_target_model
from .net_lib import train_fedavg, train_h_fedprox
from .prepare_clients import NUM_BANDS, prepare_dataset_models

SUPPORTED_ALGORITHMS = ("fedavg", "fedprox")


def _log(logger, payload):
    if logger is not None:
        logger.log(payload)


def _write_metrics(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(config, logger=None):
    """Run federated training.

    ``config`` keys (see train.py for defaults): num_clients, device, num_classes,
    algorithm, config_file, dir, data_folder, label_folder, server_rounds, epochs,
    csv_file, num_rdnn_layers, h_lr, client_lr, mu, backbone, use_hypernet,
    all_bands, batch_size, num_workers, pretrained, client_optimizer, seed,
    out_dir, hn_grad_clip.
    ``logger`` is an optional object with a ``.log(dict)`` method (e.g. wandb).
    """
    device = torch.device(config["device"])
    set_seed(config.get("seed"))

    algorithm = config["algorithm"].lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {SUPPORTED_ALGORITHMS}, got '{algorithm}'")
    backbone = config.get("backbone", "enet")
    use_hn = bool(config.get("use_hypernet", True))
    out_dir = config.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)

    # ---- clients -----------------------------------------------------------
    client_dicts = prepare_dataset_models(config)
    num_clients = len(client_dicts)
    all_channels = [client_dicts[i]["channels"] for i in range(num_clients)]

    if use_hn:
        global_channels = list(range(NUM_BANDS))
    else:
        if any(ch != all_channels[0] for ch in all_channels):
            raise ValueError(
                "Without the hypernetwork every client must hold the same bands "
                "(a fixed-input CNN cannot be aggregated across different input sizes). "
                "Use --all_bands for the 25-band vanilla baseline or pass a config where all clients match."
            )
        global_channels = list(all_channels[0])

    # ---- common validation / test splits (shared by all clients) -----------
    _, val_dataset, test_dataset = get_datasets(
        config["dir"], config["data_folder"], config["label_folder"],
        weights=[0.5, 0.3, 0.2], normMaxValue=[255, 255, 255], normMinValue=[0, 0, 0],
        test_size=16, val_size=16, client_id=1, csv_file=config.get("csv_file"), backbone=backbone,
    )
    if global_channels != list(range(NUM_BANDS)):
        val_dataset = split_bands(global_channels, val_dataset)
        test_dataset = split_bands(global_channels, test_dataset)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Common val images: {len(val_dataset)}, common test images: {len(test_dataset)}")

    # ---- models -------------------------------------------------------------
    num_classes = config["num_classes"]
    D = make_dice(num_classes, device)

    global_model = get_target_model(backbone, num_classes, device, in_channels=len(global_channels), pretrained=config.get("pretrained", True))
    replace_batchnorm(device, global_model)
    replace_prelu(device, global_model)
    for c in range(num_clients):
        replace_batchnorm(device, client_dicts[c]["model"])
        replace_prelu(device, client_dicts[c]["model"])

    hypernet, hn_keys, h_optimizer = None, [], None
    if use_hn:
        hypernet = get_hypernet(backbone, device, num_hn_layers=config.get("num_rdnn_layers", 0))
        hn_keys = hypernet_keys(hypernet, device)
        h_optimizer = optim.Adam(hypernet.parameters(), lr=config["h_lr"])
        print(f"Hypernetwork generates {len(hn_keys)} tensors: {hn_keys[:4]} ...")
    hn_key_set = set(hn_keys)

    aggregator = getattr(importlib.import_module(f"utils.algorithms.{algorithm}"), algorithm)()
    print(f"Backbone: {backbone} | algorithm: {aggregator.algorithm} | hypernetwork: {use_hn} | clients: {num_clients}")

    # global shared weights start from client 0's initialisation (as in the original runs)
    global_weights = {k: v for k, v in client_dicts[0]["model"].state_dict().items() if k not in hn_key_set}

    client_optimizer = config.get("client_optimizer")  # None -> adam for fedavg, sgd for fedprox
    rounds = config["server_rounds"]
    best_val, history = -1.0, []

    for r in tqdm(range(rounds), desc="server rounds"):
        generated, trained = [], []
        if hypernet is not None:
            hypernet.train()

        for i in range(num_clients):
            client = client_dicts[i]
            model = client["model"]
            model.load_state_dict(global_weights, strict=False)

            if hypernet is not None:
                mask = client["in_channels"].to(device)
                l1 = OrderedDict(hypernet(mask))
                load_weights_sd(model, l1)  # no graph into the client model
                generated.append(l1)

            if algorithm == "fedprox":
                model = train_h_fedprox(model, client["train_loader"], client["val_loader"], config["epochs"],
                                        device=device, lr=config["client_lr"], mu=config["mu"], optimizer_name=client_optimizer)
            else:
                model = train_fedavg(model, client["train_loader"], client["val_loader"], config["epochs"],
                                     device=device, lr=config["client_lr"], optimizer_name=client_optimizer)
            client["model"] = model

            if hypernet is not None:
                sd = model.state_dict()
                trained.append({k: sd[k].detach().clone() for k in generated[-1].keys()})

        # ---- server: aggregate shared layers ----------------------------------
        state_dicts = [{k: v for k, v in client_dicts[i]["model"].state_dict().items() if k not in hn_key_set} for i in range(num_clients)]
        global_weights = aggregator.aggregate(global_weights, state_dicts, exclude_keys=hn_keys)

        # ---- server: hypernetwork step ----------------------------------------
        hn_loss_value = None
        if hypernet is not None:
            h_optimizer.zero_grad()
            loss = hypernet_loss(generated, trained)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), config.get("hn_grad_clip", 2.0))
            h_optimizer.step()
            h_optimizer.zero_grad()
            hn_loss_value = float(loss.item())
            hypernet.eval()
            with torch.no_grad():
                l1_global = OrderedDict(hypernet(torch.ones((1, NUM_BANDS), dtype=torch.float32, device=device)))

        # ---- global model for evaluation --------------------------------------
        global_model.load_state_dict(global_weights, strict=False)
        if hypernet is not None:
            load_weights_sd(global_model, l1_global)

        test_dice = evaluate(global_model, test_loader, D, device)
        val_dice = evaluate(global_model, val_loader, D, device)
        row = {"round": r + 1, "val_dice": val_dice, "test_dice": test_dice, "hn_loss": hn_loss_value}
        history.append(row)
        _log(logger, {"epoch": r + 1, "val_dice": val_dice, "test_dice": test_dice, "hn_loss": hn_loss_value})
        print(f"Round {r + 1}/{rounds}: global val Dice {val_dice:.4f} | global test Dice {test_dice:.4f}"
              + (f" | HN loss {hn_loss_value:.4f}" if hn_loss_value is not None else ""))

        # checkpoints: best by validation Dice, plus the last round
        torch.save(global_model.state_dict(), os.path.join(out_dir, "last_global.pth"))
        if hypernet is not None:
            torch.save(hypernet.state_dict(), os.path.join(out_dir, "last_hypernet.pth"))
        if val_dice > best_val:
            best_val = val_dice
            torch.save(global_model.state_dict(), os.path.join(out_dir, "best_global.pth"))
            if hypernet is not None:
                torch.save(hypernet.state_dict(), os.path.join(out_dir, "best_hypernet.pth"))
            print(f"  new best validation Dice ({best_val:.4f}); checkpoints saved to {out_dir}")
        _write_metrics(os.path.join(out_dir, "metrics.csv"), history)

        # next round starts from the aggregated global model
        global_weights = {k: v for k, v in global_model.state_dict().items() if k not in hn_key_set}

    print(f"Done. Best global validation Dice: {best_val:.4f}. Metrics: {os.path.join(out_dir, 'metrics.csv')}")
    return {"best_val_dice": best_val, "history": history}
