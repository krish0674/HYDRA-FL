import os
import scipy.io
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "updated_dataset_distribution.csv")

# Input size per backbone, matching the downsampling factors each network needs
# (paper, Section V-B): ENet 408 x 216, UNet (ResNet-18 encoder) 416 x 224.
RESIZE = {"enet": (408, 216), "unet": (416, 224)}


def make_transform(backbone="enet"):
    h, w = RESIZE[backbone]
    return A.Compose([A.Resize(h, w)])


a_transform = make_transform("enet")

class HSIDriveDataset(Dataset):
    def __init__(self, dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, mode='train', client_id=None, implicit_norm=False, usePatches=True, csv_file=None, backbone='enet'):
        self.dirs = dirs
        self.implicit_norm = implicit_norm
        self.normMaxValue = torch.tensor(normMaxValue)
        self.normMinValue = torch.tensor(normMinValue)
        self.mode = mode
        self.client_id = client_id

        # Load CSV (image -> split assignment: train_1..train_10, val, test)
        if csv_file is None:
            csv_file = DEFAULT_CSV
        df = pd.read_csv(csv_file)
        
        # Filter based on mode
        if mode not in ['train', 'val', 'test', 'global']:
            raise ValueError("Mode should be either 'train', 'val', 'global' or 'test'")
        
        if mode == 'train' and client_id is not None:
            df = df[df['Set'] == f'train_{client_id}']
        elif mode == 'global':
            df_train = df[df['Set'].str.startswith('train')]
            df_filtered = pd.concat([df_train, df[df['Set'].isin(['val', 'test'])]])
            df = df_filtered.copy() 
        else:
            df = df[df['Set'] == mode]
        
        # Correcting the file paths for data and labels
        self.data_path = [os.path.join(dirs, data_folder, img.replace('.png', '_MF_TC_N_fl32.mat')) for img in df['Image']]
        self.label_path = [os.path.join(dirs, label_folder, img) for img in df['Labels']]

        self.transform = make_transform(backbone)
        if len(self.data_path) != len(self.label_path):
            raise Exception("The number of data samples is not equal to the number of labels.")
        self.weights = torch.tensor(weights)
        
        if usePatches:
            print(f'HSI-Drive dataset with {len(self.data_path)} patches')
        else:
            print(f'HSI-Drive dataset with {len(self.data_path)} images')

    def __len__(self):
        return len(self.data_path)

    @staticmethod
    def get_training_augmentation(**kwargs):
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
        return A.Compose(train_transform, is_check_shapes=False)

    def resize_nparray(self,nparray, target_height, target_width):
        resized_channels = []
        for channel in range(nparray.shape[0]):
            resized_channel = cv2.resize(nparray[channel, :, :], (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            resized_channels.append(resized_channel)
        resized_array = np.stack(resized_channels, axis=0)
        return resized_array

    def __getitem__(self, index):
        if self.implicit_norm:
            img = np.load(self.data_path[index])
        else:
            mat = scipy.io.loadmat(self.data_path[index])['cube_fl32']
            mat = mat.astype(np.float32)
            label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)
            # print(mat.shape)
            if self.mode == 'train':
                transform = self.transform(image=mat.transpose(1,2,0), mask=label)
                label = transform['mask']
                mat = transform['image']
                # print(mat.shape)
                # augmentation = self.get_training_augmentation()
                # augmented = augmentation(image=mat, mask=label)
                # mat, label = augmented['image'], augmented['mask']
            else:
                transform = self.transform(image=mat.transpose(1,2,0), mask=label)
                label = transform['mask']
                mat = transform['image']

            label = torch.tensor(label, dtype=torch.float32)
            img = self.normalize_image_npy(mat, "other", self.normMaxValue, self.normMinValue)
            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute(2,0,1)

        return img, label

    def normalize_image_npy(self, mat, method, max_values, min_values):
        num_channels = max_values.shape[0]
        max_values_np = max_values.numpy()
        min_values_np = min_values.numpy() 

        if method == 'removeMean':
            for i in range(num_channels):
                mat[:, :, i] = mat[:, :, i] - np.mean(mat[:, :, i])
                mat[:, :, i] = 2 * ((mat[:, :, i] - min_values[i]) / (max_values[i] - min_values[i])) - 1
        else:
            for i in range(num_channels):
                mat[:, :, i] = 2 * ((mat[:, :, i] - min_values_np[i]) / (max_values_np[i] - min_values_np[i])) - 1

        return mat
