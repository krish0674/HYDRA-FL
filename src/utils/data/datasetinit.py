import torch
from torch.utils.data import DataLoader

from utils.data.dataset import HSIDriveDataset


def get_datasets(dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, test_size, val_size, client_id, csv_file=None, backbone='enet'):
    """Return (train, val, test) datasets. ``train`` is the partition ``train_<client_id>``
    of the CSV; ``val`` and ``test`` are the common validation / test splits shared by
    every client. ``test_size`` / ``val_size`` are kept for API compatibility; the
    split is defined entirely by the CSV."""
    train_dataset = HSIDriveDataset(dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, mode='train', client_id=client_id, csv_file=csv_file, backbone=backbone)
    val_dataset = HSIDriveDataset(dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, mode='val', client_id=client_id, csv_file=csv_file, backbone=backbone)
    test_dataset = HSIDriveDataset(dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, mode='test', client_id=client_id, csv_file=csv_file, backbone=backbone)
    return train_dataset, val_dataset, test_dataset


def split_bands(channels, dataset):
    """Restrict every cube of ``dataset`` to the band indices in ``channels``
    (materialised in memory, as in the original runs)."""
    final_list = []
    idx = torch.tensor(channels)
    for img, label in dataset:
        selected_channels_img = torch.index_select(img, 0, idx)
        final_list.append((selected_channels_img, label))
    return final_list


def get_loaders(channels, n_nodes, dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, test_size, val_size, csv_file=None, batch_size=8, num_workers=4, backbone='enet'):
    """Per-client loaders. Client i trains on partition ``train_<i+1>`` restricted to its
    bands; its val / test loaders are the common splits restricted to the same bands."""
    train_loaders, test_loaders, val_loaders = [], [], []
    for i in range(n_nodes):
        train_dataset, val_dataset, test_dataset = get_datasets(dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, test_size, val_size, client_id=(1 + i), csv_file=csv_file, backbone=backbone)
        train_dataset = split_bands(channels[i], train_dataset)
        test_dataset = split_bands(channels[i], test_dataset)
        val_dataset = split_bands(channels[i], val_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        train_loaders.append(train_dataloader)
        test_loaders.append(test_dataloader)
        val_loaders.append(val_dataloader)
    return train_loaders, test_loaders, val_loaders
