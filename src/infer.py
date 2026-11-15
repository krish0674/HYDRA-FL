"""Evaluate trained HYDRA-FL (or vanilla FL) checkpoints.

Reports, on the common test split (and optionally the common validation split):
  * the global model with all 25 bands ("Global Test" in the paper), and
  * one score per client, where the client's model is the global shared layers plus
    the hypernetwork-generated initial layers for that client's band mask, applied
    to the common split restricted to the client's bands (Table IV in the paper).

Example (run from ``src``):
  python infer.py --backbone enet --num_clients 10 --num_hn_layers 3 \
      --dir /path/to/HSI-Drive --data_folder cubes_float32 --label_folder labels \
      --global_ckpt runs/enet_fedavg/best_global.pth --hypernet_ckpt runs/enet_fedavg/best_hypernet.pth
"""

import argparse
import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from utils.data.datasetinit import get_datasets
from utils.hn_utils import evaluate, hypernet_keys, load_weights_sd, make_dice, replace_batchnorm, replace_prelu
from utils.models.model import get_hypernet, get_target_model
from utils.prepare_clients import NUM_BANDS, prepare_dataset_models

HERE = os.path.dirname(os.path.abspath(__file__))


def build_parser():
    p = argparse.ArgumentParser(description="HYDRA-FL inference / client-wise evaluation")
    p.add_argument("--dir", type=str, default="")
    p.add_argument("--data_folder", type=str, default="data_folder")
    p.add_argument("--label_folder", type=str, default="label_folder")
    p.add_argument("--csv_file", type=str, default=os.path.join(HERE, "utils", "dataset", "updated_dataset_distribution.csv"))
    p.add_argument("--config_file", type=str, default=os.path.join(HERE, "config.json"))
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--backbone", type=str, default="enet", choices=["enet", "unet"])
    p.add_argument("--num_hn_layers", "--num_rdnn_layers", dest="num_hn_layers", type=int, default=0)
    p.add_argument("--all_bands", action="store_true")
    p.add_argument("--global_ckpt", "--e_path", dest="global_ckpt", type=str, required=True, help="Checkpoint of the global model (best_global.pth)")
    p.add_argument("--hypernet_ckpt", "--h_path", dest="hypernet_ckpt", type=str, default=None, help="Checkpoint of the hypernetwork (omit for a vanilla FL model)")
    p.add_argument("--split", type=str, default="test", choices=["test", "val", "both"], help="Which common split to score")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default=None, help="Optional path to write the scores as JSON")
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)
    config = {
        "num_clients": args.num_clients, "device": device, "num_classes": args.num_classes,
        "config_file": args.config_file, "dir": args.dir, "data_folder": args.data_folder,
        "label_folder": args.label_folder, "csv_file": args.csv_file, "backbone": args.backbone,
        "all_bands": args.all_bands, "batch_size": 1, "num_workers": args.num_workers,  # batch 1: per-image Dice as in training
        "pretrained": False,
    }
    client_dicts = prepare_dataset_models(config)
    D = make_dice(args.num_classes, device)

    # global model (all 25 bands) on the common split(s)
    global_model = get_target_model(args.backbone, args.num_classes, device, in_channels=NUM_BANDS, pretrained=False)
    replace_batchnorm(device, global_model)
    replace_prelu(device, global_model)
    global_sd = torch.load(args.global_ckpt, map_location=device)
    missing, unexpected = global_model.load_state_dict(global_sd, strict=False)
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected keys in the global checkpoint, e.g. {unexpected[:3]}")

    hypernet, hn_keys = None, set()
    if args.hypernet_ckpt:
        hypernet = get_hypernet(args.backbone, device, num_hn_layers=args.num_hn_layers)
        hypernet.load_state_dict(torch.load(args.hypernet_ckpt, map_location=device))
        hypernet.eval()
        hn_keys = set(hypernet_keys(hypernet, device))
        with torch.no_grad():
            load_weights_sd(global_model, OrderedDict(hypernet(torch.ones((1, NUM_BANDS), device=device))))

    _, val_ds, test_ds = get_datasets(args.dir, args.data_folder, args.label_folder, weights=[0.5, 0.3, 0.2],
                                      normMaxValue=[255, 255, 255], normMinValue=[0, 0, 0], test_size=16, val_size=16,
                                      client_id=1, csv_file=args.csv_file, backbone=args.backbone)
    splits = {"test": test_ds, "val": val_ds} if args.split == "both" else {args.split: test_ds if args.split == "test" else val_ds}

    results = {}
    for split_name, ds in splits.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        results[f"global_{split_name}_dice"] = evaluate(global_model, loader, D, device)
        print(f"Global model, all {NUM_BANDS} bands, common {split_name} split: Dice {results[f'global_{split_name}_dice']:.4f}")

    # client-wise: shared layers from the global model, initial layers from the hypernetwork
    shared = {k: v for k, v in global_model.state_dict().items() if k not in hn_keys}
    for i in range(args.num_clients):
        client = client_dicts[i]
        model = client["model"]
        replace_batchnorm(device, model)
        replace_prelu(device, model)
        model.load_state_dict(shared, strict=False)
        if hypernet is not None:
            with torch.no_grad():
                load_weights_sd(model, OrderedDict(hypernet(client["in_channels"].to(device))))
        for split_name in splits:
            loader = client["test_loader"] if split_name == "test" else client["val_loader"]
            score = evaluate(model, loader, D, device)
            results[f"client_{i + 1}_{split_name}_dice"] = score
            print(f"Client {i + 1:2d} ({len(client['channels']):2d} bands), common {split_name} split: Dice {score:.4f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Scores written to {args.out_json}")


if __name__ == "__main__":
    main()
