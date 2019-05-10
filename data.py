import torch
import numpy as np
from torch.utils.data import Dataset


class PendulumDataset(Dataset):
    def __init__(self, dataset_type=None):
        if dataset_type == 'train':
            data = np.load('./pend_data/pendulum_no_action_train.npy')
            self.data = data.astype(np.float32)
        elif dataset_type == 'valid':
            data = np.load('./pend_data/pendulum_no_action_valid.npy')
            self.data = data.astype(np.float32)
        elif dataset_type == 'test':
            data = np.load('./pend_data/pendulum_no_action_valid.npy')
            self.data = data.astype(np.float32)
        else:
            raise NotImplementedError

        self.state_size = self.data.shape[2]-1

    def __getitem__(self, index):
        traj_i = self.data[index]

        states = torch.from_numpy(traj_i[:, 0])
        actions = torch.from_numpy(traj_i[:, -1])

        return states, actions

    def __len__(self):
        return self.data.shape[0]
