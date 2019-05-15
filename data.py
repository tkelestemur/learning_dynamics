import torch
import numpy as np
from torch.utils.data import Dataset


class PendulumDataset(Dataset):
    def __init__(self, dataset_type=None):
        if dataset_type == 'train':
            data = np.load('./pend_data/pendulum_train_bounded.npy')
        elif dataset_type == 'valid':
            data = np.load('./pend_data/pendulum_valid_bounded.npy')
        elif dataset_type == 'test':
            data = np.load('./pend_data/pendulum_valid_bounded.npy')
        else:
            raise NotImplementedError

        self.data = data.astype(np.float32)
        # self.data_normalized = np.empty_like(data, dtype=np.float32)
        # self.data_normalized[:, 0] = (data[:, 0] - data[:, 0].min()) / (data[:, 0].max() - data[:, 0].min())
        # self.data_normalized[:, 1] = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max() - data[:, 1].min())
        # self.data_normalized[:, 2] = (data[:, 2] - data[:, 2].min()) / (data[:, 2].max() - data[:, 2].min())
        # self.data_normalized[:, 3] = (data[:, 3] - data[:, 3].min()) / (data[:, 3].max() - data[:, 3].min())
        # self.data_normalized = self.data_normalized.reshape(data.shape[0], 1)
        # self.data_normalized = self.data_normalized.astype(np.float32)
        # self.data[:, 0] = (data[:, 0] - data[:, 0].min()) / (data[:, 0].max() - data[:, 0].min())
        # self.data[:, 1] = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max() - data[:, 1].min())
        # self.data[:, 2] = (data[:, 2] - data[:, 2].min()) / (data[:, 2].max() - data[:, 2].min())
        # self.data[:, 3] = (data[:, 3] - data[:, 3].min()) / (data[:, 3].max() - data[:, 3].min())


    def __getitem__(self, index):
        # traj_i = self.data[index]
        states = torch.from_numpy(self.data[index])
        return states

    def __len__(self):
        return self.data.shape[0]
