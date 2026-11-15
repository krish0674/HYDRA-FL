"""Train HYDRA-FL (hypernetwork-generated input layers + FedAvg/FedProx) or a
vanilla FedAvg/FedProx baseline on HSI-Drive.

Examples (run from the ``src`` directory):

  # HYDRA-FL, ENet backbone, FedAvg, 10 clients with the band split of config.json
  python train.py --backbone enet --algorithm fedavg --num_clients 10 \
      --dir /path/to/HSI-Drive --data_folder cubes_float32 --label_folder labels \
      --server_rounds 50 --client_epochs 40 --h_lr 0.1 --client_lr 0.01 --num_hn_layers 3

  # same with FedProx
  python train.py --backbone enet --algorithm fedprox --mu 0.001 ... (other flags as above)

  # vanilla FedAvg baseline: no hypernetwork, every client sees all 25 bands
  python train.py --backbone enet --algorithm fedavg --no_hypernet --all_bands ...
"""

import argparse
import json
import os

import torch

from utils.trainer import train

HERE = os.path.dirname(os.path.abspath(__file__))


def build_parser():
    p = argparse.ArgumentParser(description="HYDRA-FL / vanilla FL training on HSI-Drive")
    # data
    p.add_argument("--dir", type=str, default="", help="Root directory of the dataset")
    p.add_argument("--data_folder", type=str, default="data_folder", help="Sub-folder with the *_MF_TC_N_fl32.mat cubes")
    p.add_argument("--label_folder", type=str, default="label_folder", help="Sub-folder with the label PNGs")
    p.add_argument("--csv_file", type=str, default=os.path.join(HERE, "utils", "dataset", "updated_dataset_distribution.csv"),
                   help="CSV assigning every image to train_<client>, val or test")
    p.add_argument("--config_file", type=str, default=os.path.join(HERE, "config.json"), help="JSON with the band indices held by each client")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--num_classes", type=int, default=11, help="10 HSI-Drive classes + background")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    # method
    p.add_argument("--backbone", type=str, default="enet", choices=["enet", "unet"])
    p.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg", "fedprox"])
    p.add_argument("--no_hypernet", action="store_true", help="Vanilla FL: no hypernetwork, all layers aggregated (clients must share the same bands)")
    p.add_argument("--all_bands", action="store_true", help="Give every client all 25 bands (ignores --config_file)")
    p.add_argument("--num_hn_layers", "--num_rdnn_layers", dest="num_hn_layers", type=int, default=0,
                   help="Number of blocks after the initial block that the hypernetwork also generates (ENet: RDDN bottlenecks, UNet: encoder stages)")
    p.add_argument("--no_pretrained", action="store_true", help="UNet only: do not load ImageNet encoder weights")
    # optimisation
    p.add_argument("--server_rounds", type=int, default=10)
    p.add_argument("--client_epochs", type=int, default=1)
    p.add_argument("--h_lr", type=float, default=0.001, help="Hypernetwork learning rate (Adam)")
    p.add_argument("--client_lr", type=float, default=0.001, help="Client learning rate")
    p.add_argument("--client_optimizer", type=str, default=None, choices=[None, "adam", "sgd"],
                   help="Override the client optimiser (default: adam for fedavg, sgd+momentum for fedprox, as in the original runs)")
    p.add_argument("--mu", type=float, default=0.001, help="FedProx proximal coefficient")
    p.add_argument("--hn_grad_clip", type=float, default=2.0, help="Max gradient norm for the hypernetwork update")
    p.add_argument("--seed", type=int, default=None)
    # runtime / logging
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="runs/default", help="Where checkpoints and metrics.csv are written")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (project/entity from --wandb_project/--wandb_entity)")
    p.add_argument("--wandb_project", type=str, default="HYDRA-FL")
    p.add_argument("--wandb_entity", type=str, default=None)
    return p


def config_from_args(args):
    return {
        "num_clients": args.num_clients,
        "device": args.device,
        "num_classes": args.num_classes,
        "algorithm": args.algorithm,
        "config_file": args.config_file,
        "dir": args.dir,
        "data_folder": args.data_folder,
        "label_folder": args.label_folder,
        "server_rounds": args.server_rounds,
        "epochs": args.client_epochs,
        "csv_file": args.csv_file,
        "num_rdnn_layers": args.num_hn_layers,
        "h_lr": args.h_lr,
        "client_lr": args.client_lr,
        "mu": args.mu,
        "backbone": args.backbone,
        "use_hypernet": not args.no_hypernet,
        "all_bands": args.all_bands,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pretrained": not args.no_pretrained,
        "client_optimizer": args.client_optimizer,
        "hn_grad_clip": args.hn_grad_clip,
        "seed": args.seed,
        "out_dir": args.out_dir,
    }


def main():
    args = build_parser().parse_args()
    config = config_from_args(args)
    os.makedirs(config["out_dir"], exist_ok=True)
    with open(os.path.join(config["out_dir"], "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger = None
    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config)
        logger = wandb

    train(config, logger=logger)


if __name__ == "__main__":
    main()
