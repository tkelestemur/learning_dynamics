import torch
import numpy as np
from torch.utils.data import Dataset


class PendulumDataset(Dataset):
    def __init__(self, dataset_type=None):
        if dataset_type == 'train':
            data = np.load('./pend_data/pendulum_no_action_bounded_train.npy')
        elif dataset_type == 'valid':
            data = np.load('./pend_data/pendulum_no_action_bounded_valid.npy')

        self.data = data.astype(np.float32)

    def normalize(self, index):
        data_norm = (self.data[index] + 1) / 2
        return data_norm

    def __getitem__(self, index):
        states = torch.from_numpy(self.data[index])
        return states

    def __len__(self):
        return self.data.shape[0]
