# HYDRA-FL: Hypernet-Driven Aggregation for Resilient Federated Learning in Band Heterogeneous Environments

Code for the paper *HYDRA-FL: Hypernet-Driven Aggregation for Resilient Federated Learning in Band Heterogeneous Environments* (Krish Didwania, Aditya Kasliwal, Ishaan Gakhar, Annappa B, Sachin D N; submitted to the Federated HPC workshop at IEEE HiPC 2026).

HYDRA-FL trains one segmentation model across federated clients whose hyperspectral cameras capture **different subsets of the spectral bands**, so their input layers have different shapes and cannot be averaged. A server-side HyperNetwork, conditioned on each client's binary band-presence mask, generates the band-dependent initial layers of that client's model; the remaining layers, identical for every client, are aggregated with FedAvg or FedProx. Raw images and band configurations never leave the client.

The same `train.py` also runs the **vanilla FedAvg / FedProx baselines** (no HyperNetwork, every client with the same bands), which are the "Band Heterogeneity = False" rows of the paper.

## Repository layout

```
src/
  train.py                  federated training (HYDRA-FL or vanilla FL)
  infer.py                  global and client-wise evaluation of saved checkpoints
  config.json               band indices held by each of the 10 clients (Table II of the paper)
  utils/
    trainer.py              the server loop (Algorithm 1 of the paper)
    hn_utils.py             weight loading, hypernetwork loss, Dice metric, norm/activation swaps
    prepare_clients.py      builds one model + data loaders + band mask per client
    net_lib.py              client-side training (FedAvg: Adam; FedProx: SGD + proximal term)
    algorithms/             server aggregation rules (fedavg.py, fedprox.py)
    models/ENet.py          ENet client model and EnetHyper1 hypernetwork
    models/UNetHyper.py     hypernetwork for the UNet (ResNet-18 encoder) client model
    data/dataset.py         HSI-Drive loader (.mat cubes + PNG labels, resize, normalisation)
    dataset/updated_dataset_distribution.csv   image -> split assignment (train_1..train_10, val, test)
requirements.txt
```

## Installation

Python 3.10 or newer, PyTorch 2.x. The reported runs used Python 3.10.13, torch 2.1.2, torchmetrics 1.3.2, albumentations 1.4.0 and segmentation-models-pytorch 0.3.4 on a Kaggle Tesla P100 (16 GB).

```bash
git clone https://github.com/krish0674/HYDRA-FL.git
cd HYDRA-FL
pip install -r requirements.txt
# UNet backbone only:
pip install segmentation-models-pytorch
```

## Data: HSI-Drive v1

The experiments use the HSI-Drive dataset (Basterretxea et al., IEEE IV 2021): per-pixel annotated 25-band VNIR images recorded from a car. It is freely available for non-commercial use from the authors' page, https://ipaccess.ehu.eus/HSI-Drive/ (the archive password is obtained by e-mail as described there). The paper uses the 276 images of version 1 with the 10 annotated classes plus background (11 classes).

The loader expects two folders, referenced as `--dir <root> --data_folder <cubes> --label_folder <labels>` (the two folders may also be given as absolute paths with `--dir ""`):

```
<root>/<cubes>/<image>_MF_TC_N_fl32.mat     float32 cube, MATLAB key 'cube_fl32', shape 25 x H x W
<root>/<labels>/<image>.png                 8-bit label map with class ids 0..10
```

`src/utils/dataset/updated_dataset_distribution.csv` lists the 276 image names with their split: `train_1` ... `train_10` (the training images of the ten clients, 24 or 25 each, 244 in total), `val` (16 images) and `test` (16 images). The validation and test images are shared by all clients. Images are resized to 408 x 216 (ENet) or 416 x 224 (UNet).

### Band split across clients

`src/config.json` gives the band indices (0-24) held by each client. This is the split used for every HYDRA-FL result in the paper:

| client (paper) | key in config.json | bands | indices |
|---|---|---|---|
| 1  | client_0 | 11 | 0-10 |
| 2  | client_1 | 10 | 7-16 |
| 3  | client_2 | 25 | all |
| 4  | client_3 | 10 | 0,1,2,3,4,18,19,20,21,22 |
| 5  | client_4 | 10 | 0,1,2,14,15,16,17,18,23,24 |
| 6  | client_5 | 16 | 3-13, 19-23 |
| 7  | client_6 | 10 | 8-13, 20,21,22,24 |
| 8  | client_7 | 10 | 9-18 |
| 9  | client_8 | 15 | 5-11, 15,16,17, 19-23 |
| 10 | client_9 | 25 | all |

Any other split can be used by editing the file or passing another JSON with `--config_file`.

## Running

All commands are run from `src/`. The settings below are the ones used for the paper (Section V-B): 10 clients, 50 server rounds, 40 local epochs per round, batch size 8, HyperNetwork learning rate 0.1 (Adam, gradient norm clipped at 2), client learning rate 0.01, cross-entropy loss, and the HyperNetwork generating the initial block plus the first 3 blocks (`--num_hn_layers 3`: RDDN bottlenecks for ENet, encoder stages for UNet).

HYDRA-FL, ENet backbone, FedAvg:

```bash
python train.py --backbone enet --algorithm fedavg \
  --dir /data/HSI-Drive --data_folder cubes_float32 --label_folder labels \
  --num_clients 10 --server_rounds 50 --client_epochs 40 --batch_size 8 \
  --h_lr 0.1 --client_lr 0.01 --num_hn_layers 3 \
  --out_dir runs/enet_fedavg
```

HYDRA-FL with FedProx (clients use SGD with momentum 0.9 plus the proximal term; `--mu` is the proximal coefficient):

```bash
python train.py --backbone enet --algorithm fedprox --mu 0.001 \
  --dir /data/HSI-Drive --data_folder cubes_float32 --label_folder labels \
  --num_clients 10 --server_rounds 50 --client_epochs 40 --batch_size 8 \
  --h_lr 0.1 --client_lr 0.01 --num_hn_layers 3 \
  --out_dir runs/enet_fedprox
```

UNet backbone (ResNet-18 encoder from segmentation-models-pytorch, ImageNet-initialised; add `--no_pretrained` to skip the download):

```bash
python train.py --backbone unet --algorithm fedavg \
  --dir /data/HSI-Drive --data_folder cubes_float32 --label_folder labels \
  --num_clients 10 --server_rounds 50 --client_epochs 40 --batch_size 8 \
  --h_lr 0.1 --client_lr 0.01 --num_hn_layers 3 \
  --out_dir runs/unet_fedavg
```

Vanilla FedAvg / FedProx baselines (no HyperNetwork, every client sees all 25 bands, all layers averaged):

```bash
python train.py --backbone enet --algorithm fedavg --no_hypernet --all_bands \
  --dir /data/HSI-Drive --data_folder cubes_float32 --label_folder labels \
  --num_clients 10 --server_rounds 50 --client_epochs 40 --batch_size 8 --client_lr 0.01 \
  --out_dir runs/enet_fedavg_vanilla
```

Without the HyperNetwork all clients must hold the same bands (a fixed-input CNN cannot be averaged across different input sizes); the script stops with a clear error otherwise.

Other useful flags: `--seed`, `--device cpu|cuda`, `--client_optimizer adam|sgd` (override the per-algorithm default), `--num_workers`, `--wandb --wandb_project NAME --wandb_entity NAME` for Weights & Biases logging. `python train.py -h` lists everything.

### Outputs

Each run writes to `--out_dir`:

* `metrics.csv`: per round, Dice of the global model on the common validation and test splits, and the HyperNetwork loss;
* `best_global.pth` / `best_hypernet.pth`: checkpoints at the round with the best validation Dice;
* `last_global.pth` / `last_hypernet.pth`: checkpoints after the final round;
* `run_config.json`: the full configuration.

### Evaluation (Table IV of the paper)

`infer.py` scores the global model with all 25 bands and, for every client, the model made of the global shared layers plus the HyperNetwork-generated initial layers for that client's band mask, on the common split restricted to the client's bands:

```bash
python infer.py --backbone enet --num_clients 10 --num_hn_layers 3 \
  --dir /data/HSI-Drive --data_folder cubes_float32 --label_folder labels \
  --global_ckpt runs/enet_fedavg/best_global.pth --hypernet_ckpt runs/enet_fedavg/best_hypernet.pth \
  --split test --out_json runs/enet_fedavg/scores.json
```

For a vanilla model omit `--hypernet_ckpt` and pass `--all_bands`. `--split both` scores the validation split as well.

## How the HyperNetwork handles different band counts

`EnetHyper1` (and `UnetHyper`) emit the first convolution for a full 25-band input, for example a 13 x 25 x 3 x 3 kernel for ENet, and then keep only the input-channel slices whose band is present in the client's mask (`w1[:, one_hot_indices]`). The generated tensor therefore has exactly the client's band count, and the training loss compares each generated tensor with the locally trained tensor of the *same* client, so tensors of different clients are never compared with each other. The locally trained weights enter the loss as constants; the gradient flows through the HyperNetwork only.

## Results reported in the paper (single runs)

Global test Dice (mean over the 16 common test images, micro-averaged over the 11 classes):

| Backbone | Algorithm | Vanilla (all 25 bands at every client) | HYDRA-FL (band split of Table II) |
|---|---|---|---|
| ENet | FedAvg  | 0.786 | 0.811 |
| ENet | FedProx | 0.769 | 0.798 |
| UNet | FedAvg  | 0.793 | 0.817 |
| UNet | FedProx | 0.790 | 0.810 |

The two columns are not a like-for-like comparison: the vanilla rows see all bands at every client, the HYDRA-FL rows do not. All numbers are from single runs on Kaggle (Tesla P100, 16 GB); the development notebook that produced them (`fl-hypernet-runner`, Kaggle, private to the authors) ran the code in this repository with the commands above.

## Notes on the implementation (for anyone comparing with the paper)

These are properties of the code as it was run for the paper; they are kept so the released code reproduces the reported setting.

* **Metric.** `torchmetrics` Dice with `average='micro'` over all 11 classes including background, computed per image and averaged over images. A micro average over all classes of a single-label segmentation equals the overall pixel accuracy. torchmetrics removed the `Dice` class in version 1.8; the code falls back to `MulticlassF1Score(average='micro')`, which is the same quantity.
* **Aggregation.** The server step is an unweighted mean over clients. With 24 or 25 training images per client this coincides with the `n_i / n` weighting of Eq. (1) up to that difference.
* **HyperNetwork loss.** Sum of squared differences over all generated entries and all clients (no division by the number of clients or entries); the scale is absorbed by the learning rate.
* **Client optimiser.** Adam for FedAvg clients, SGD with momentum 0.9 for FedProx clients (`--client_optimizer` overrides this).
* **Normalisation.** `dataset.py` applies the `2 * (x / 255) - 1` scaling to the first three channels only (`normMaxValue` has three entries), as in the original runs.
* **Model selection.** This release keeps the checkpoint with the best *validation* Dice; the development code selected on the test split.
* **Loader naming.** In the development code the per-client validation and test loaders were swapped (the client-wise scores were computed on the 16-image validation split). This release uses the splits as named in the CSV.
* **InstanceNorm / ReLU.** Every BatchNorm2d is replaced by an affine InstanceNorm2d and every PReLU by ReLU before training, for both backbones.

## Citation

```bibtex
@inproceedings{didwania2026hydrafl,
  title     = {{HYDRA-FL}: Hypernet-Driven Aggregation for Resilient Federated Learning in Band Heterogeneous Environments},
  author    = {Didwania, Krish and Kasliwal, Aditya and Gakhar, Ishaan and B, Annappa and D N, Sachin},
  booktitle = {Federated HPC Workshop, IEEE International Conference on High Performance Computing, Data, and Analytics (HiPC)},
  year      = {2026},
  note      = {under review}
}
```

Dataset: K. Basterretxea, V. Martinez, J. Echanobe, J. Gutierrez-Zaballa and I. del Campo, "HSI-Drive: A Dataset for the Research of Hyperspectral Image Processing Applied to Autonomous Driving Systems", IEEE Intelligent Vehicles Symposium (IV), 2021.
