import torch
import numpy as np
from torch.utils.data import Dataset


class PendulumDataset(Dataset):
    def __init__(self):
        # self.data = np.load('./pend_data/pendulum_6k_disc.npy')
        self.data = np.load('./pend_data/pendulum_6ktraj_cont_100steps.npy')

    def __getitem__(self, index):
        traj_i = self.data[index]
        return torch.Tensor(traj_i)

    def __len__(self):
        return self.data.shape[0]
