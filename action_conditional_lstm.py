from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from data import PendulumDataset
import numpy as np


class ActionCondCLSTM(nn.Module):

    def __init__(self, input_size, action_size, hidden_size, num_layers):
        super(ActionCondCLSTM, self).__init__()

        self.action_size = action_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.lstm_cell = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.f_states_enc = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.f_states_dec = nn.Linear(in_features=hidden_size, out_features=input_size)

        self.f_actions = nn.Linear(in_features=action_size, out_features=hidden_size, bias=False)
        self.f_enc = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.f_dec = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, x):

        # state input from 0th to T-1 index
        states = x[:, :, 0:3]
        action = x[:, :, 3:4]

        # encoded = F.relu(self.f_states_enc(states))
        encoded, hidden = self.lstm(states)

        # action transformation
        x_hat = self.f_dec(self.f_enc(encoded) * self.f_actions(action))

        # decoded = F.relu(self.f_states_dec(x_hat))

        return x_hat

    # def forward(self, x):
    #     # h_t = torch.zeros(x.size(0), self.hidden_size, device='cuda')
    #     # c_t = torch.zeros(x.size(0), self.hidden_size, device='cuda')
    #     h_t = torch.empty(x.size(0), self.hidden_size, device='cuda').uniform_(-0.08, 0.08)
    #     c_t = torch.empty(x.size(0), self.hidden_size, device='cuda').uniform_(-0.08, 0.08)
    #
    #     S_hat = []
    #     for x_i, x_t in enumerate(x.chunk(x.size(1), dim=1)):
    #         x_t = x_t.view(x_t.size(0), x_t.size(2))
    #         # print(x_t.shape)
    #         s_t = x_t[:, 0:2]
    #         a_t = x_t[:, 3:4]
    #         # print(s_t.shape)
    #         s_t_enc = self.f_states_enc(s_t)
    #         # print(s_t_enc.shape)
    #         s_a_t = self.f_dec(self.f_enc(h_t) * self.f_actions(a_t))
    #
    #         h_t, c_t = self.lstm_cell(s_a_t, (h_t, c_t))
    #
    #
    #
    #         # print(s_a_t.shape)
    #         s_hat = self.f_states_dec(h_t)
    #
    #         S_hat += [s_hat]
    #
    #     S_hat = torch.stack(S_hat, 1)
    #     # print('Shat shape: {}'.format(S_hat.shape))
    #     return S_hat

    def train_model(self, num_epochs=1, data_loader=None, device=None):
        self.train()
        loss_func = nn.MSELoss()
        # loss_func = nn.BCELoss()
        optimizer = optim.Adam(params=self.parameters(), lr=1e-3)

        reco_loss = []
        for epoch in range(num_epochs):
            train_loss = 0
            for batch in data_loader:
                batch = batch.to(device)

                # forward pass
                optimizer.zero_grad()

                target = batch[:, :, 4:7]

                x_hat = self.forward(batch)
                # print(batch.shape)
                # print(x_hat.shape)

                # backward pass
                loss = loss_func(input=x_hat, target=target)
                loss.backward()
                train_loss += loss.item()

                # gradient update
                optimizer.step()
                # break
            # break
                # print('Batch loss: {}'.format(loss.item()))
            print('Epoch {} loss: {}'.format(epoch, train_loss / len(data_loader)))
            # reco_loss.append(train_loss / len(data_loader.dataset)

        # reco_loss = np.array(reco_loss)

        # np.savetxt('./data/loss_5k.csv', reco_loss, delimiter=',')


if __name__ == '__main__':

    # Load dataset
    pend_data = PendulumDataset()
    # pend_data = pend_dat
    pend_loader = DataLoader(dataset=pend_data, batch_size=32, drop_last=True,
                             shuffle=False, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ActionCondCLSTM(input_size=3, action_size=1, hidden_size=256, num_layers=1).to(device)
    model.train_model(num_epochs=5000, data_loader=pend_loader, device=device)
