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

        self.f_actions = nn.Linear(in_features=action_size, out_features=hidden_size)
        self.f_enc = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.f_dec = nn.Linear(in_features=2*hidden_size, out_features=input_size)

    def forward(self, x):

        # state input from 0th to T-1 index
        states = x[:, :, 0:3]
        action = x[:, :, 3:4]

        # encoded = F.relu(self.f_states_enc(states))
        encoded, hidden = self.lstm(states)
        # print(encoded.shape)

        # action transformation
        state_and_action = torch.cat((self.f_enc(encoded), self.f_actions(action)), dim=2)
        # print(state_and_action.shape)
        x_hat = self.f_dec(state_and_action)

        # decoded = F.relu(self.f_states_dec(x_hat))

        return x_hat

    # def forward(self, x):
    #     # h_t = torch.zeros(x.size(0), self.hidden_size, device='cuda')
    #     # c_t = torch.zeros(x.size(0), self.hidden_size, device='cuda')
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


    def evaluate_model(self, test_data_loader=None, device=torch.device('cpu')):
        # self.eval()
        loss_func = nn.MSELoss()
        valid_loss = 0
        with torch.no_grad():
            for batch in test_data_loader:
                batch = batch.to(device)
                target = batch[:, :, 4:7]
                x_hat = self.forward(batch)
                loss = loss_func(input=x_hat, target=target)

                valid_loss += loss.item()
            valid_loss /= len(test_data_loader)
        return valid_loss

    def train_model(self, num_epochs=1, train_data_loader=None, test_data_loader=None, device=None):
        self.train()
        loss_func = nn.MSELoss()
        # loss_func = nn.BCELoss()
        optimizer = optim.Adam(params=self.parameters(), lr=1e-3)

        epoch_loss = []
        for epoch in range(num_epochs):
            train_loss = 0
            for batch in train_data_loader:
                batch = batch.to(device)

                # forward pass
                optimizer.zero_grad()
                target = batch[:, :, 4:7]
                x_hat = self.forward(batch)

                # backward pass
                loss = loss_func(input=x_hat, target=target)
                loss.backward()
                train_loss += loss.item()

                # gradient update
                optimizer.step()

                # print('Batch loss: {}'.format(loss.item()))
            valid_loss = self.evaluate_model(test_data_loader=test_data_loader, device=device)
            train_loss  = train_loss / len(train_data_loader)

            print('Epoch {} Train Loss: {} Validation Loss: {}'.format(epoch, train_loss, valid_loss))
            epoch_loss += [np.hstack((train_loss, valid_loss))]

        epoch_loss = np.array(epoch_loss)

        np.savetxt('./loss/loss_5k.csv', epoch_loss, delimiter=',')


if __name__ == '__main__':

    # Load dataset
    pend_train_data = PendulumDataset('train')
    pend_test_data = PendulumDataset('test')

    pend_train_loader = DataLoader(dataset=pend_train_data, batch_size=8, drop_last=True,
                                   shuffle=False, num_workers=4)

    pend_train_loader = DataLoader(dataset=pend_test_data, batch_size=len(pend_test_data),
                                   drop_last=False, shuffle=False, num_workers=2)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ActionCondCLSTM(input_size=3, action_size=1, hidden_size=128, num_layers=1).to(device)
    model.train_model(num_epochs=2000, train_data_loader=pend_train_loader, test_data_loader=pend_train_loader, device=device)
