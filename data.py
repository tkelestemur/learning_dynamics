import torch
import numpy as np
from torch.utils.data import Dataset


class PendulumDataset(Dataset):
    def __init__(self, dataset_type=None):
        if dataset_type == 'train':
            self.data = np.load('./pend_data/pendulum_100H_5000N.npy')
        elif dataset_type == 'test':
            self.data = np.load('./pend_data/pendulum_100H_1000N.npy')

    def __getitem__(self, index):
        traj_i = self.data[index]
        return torch.Tensor(traj_i)

    def __len__(self):
        return self.data.shape[0]
