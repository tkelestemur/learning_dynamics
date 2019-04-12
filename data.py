import torch
import numpy as np
from torch.utils.data import Dataset


class PendulumDataset(Dataset):
    def __init__(self, dataset_type=None):
        if dataset_type == 'train':
            data = np.load('./pend_data/pendulum_200_step_5k_run.npy')
            self.data = data.astype(np.float32)
        elif dataset_type == 'test':
            data = np.load('./pend_data/pendulum_200_step_1k_run.npy')
            self.data = data.astype(np.float32)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        traj_i = self.data[index]

        states = torch.from_numpy(traj_i[:, 0:3])
        actions = torch.from_numpy(traj_i[:, 3:4])
        # next_states = torch.from_numpy(traj_i[:, 4:7])

        return states, actions

    def __len__(self):
        return self.data.shape[0]
