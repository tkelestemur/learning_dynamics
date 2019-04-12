import torch
import torch.nn as nn
from torch import optim
import numpy as np


class ActionCondLSTM(nn.Module):

    def __init__(self, input_size, action_size, hidden_size, num_layers, future_steps=1, checkpoint_path=None, loss_path=None):
        super(ActionCondLSTM, self).__init__()

        self.action_size = action_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.future_steps = future_steps
        self.checkpoint_path = checkpoint_path
        self.loss_path = loss_path

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # self.lstm_cell = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        # self.f_states_enc = nn.Linear(in_features=input_size, out_features=hidden_size)
        # self.f_states_dec = nn.Linear(in_features=hidden_size, out_features=input_size)

        self.f_actions = nn.Linear(in_features=action_size, out_features=hidden_size)
        self.f_enc = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.f_dec = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, s, a):

        encoded, hidden = self.lstm(s)

        # action transformation
        s_next_h = self.f_enc(encoded) + self.f_actions(a)
        s_next = self.f_dec(s_next_h)
        # state_and_action = torch.cat((self.f_enc(encoded), self.f_actions(action)), dim=2)

        return s_next, s_next_h, encoded

    def train_model(self, num_epochs=1, train_data_loader=None, valid_data_loader=None,
                    device=None, save_model=False):
        self.train()

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(params=self.parameters(), lr=1e-3)

        print('Starting training...\nNumber of future step predictions: {}'.format(self.future_steps))
        epoch_loss = []
        for epoch in range(num_epochs):
            train_loss = 0
            for states, actions in train_data_loader:

                states = states.to(device)
                actions = actions.to(device)

                optimizer.zero_grad()
                # forward pass
                if self.future_steps == 1:
                    s_next, s_next_h, encoded = self.forward(states, actions)

                    loss = loss_func(input=s_next[:, :-1, ], target=states[:, 1:, :])
                    # loss = loss_func(input=s_next, target=next_states) + \
                    #        loss_func(input=s_next_h[:, :-1, :], target=encoded[:, 1:, :])
                elif self.future_steps == 2: #TODO: Fix this
                    s_next, s_next_h, encoded = self.forward(states, actions)
                    next_next_states_pred = self.forward(s_next[:, :-1, :], actions[:, 1:, :])
                    loss = 0.5 * (loss_func(input=s_next[:, :-1, ], target=states[:, 1:, :]) +
                                  loss_func(input=next_next_states_pred, target=next_states[:, 1:, :]))

                loss.backward()
                train_loss += loss.item()

                # gradient update
                optimizer.step()

            valid_loss = self.evaluate_model(valid_data_loader=valid_data_loader, device=device)
            # valid_loss = 0
            train_loss = train_loss / len(train_data_loader)

            print('Epoch {} Train Loss: {:.6f} Validation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))
            epoch_loss += [np.hstack((train_loss, valid_loss))]

            if save_model:
                if not self.checkpoint_path:
                    print('Checkpoint pat is not specified! Can\'t save the model..')
                else:
                    with open(self.checkpoint_path, 'wb') as f:
                        torch.save(self.state_dict(), f)

        if save_model:
            epoch_loss = np.array(epoch_loss)
            np.savetxt(self.loss_path, epoch_loss, delimiter=',')

    def evaluate_model(self, valid_data_loader=None, device=None):
        # self.eval()
        loss_func = nn.MSELoss()
        valid_loss = 0
        with torch.no_grad():
            for states, actions in valid_data_loader:
                states = states.to(device)
                actions = actions.to(device)

                if self.future_steps == 1:
                    states_next = states[:, 1:, :]

                    # next_states_pred = self.forward(states, actions)
                    # loss = loss_func(input=next_states_pred, target=next_states)
                    s_next, s_next_h, encoded = self.forward(states, actions)

                    loss = loss_func(input=s_next[:, :-1, ], target=states[:, 1:, :])

                    # loss = loss_func(input=s_next, target=next_states) + \
                    #        loss_func(input=s_next_h[:, :-1, :], target=encoded[:, 1:, :])

                elif self.future_steps == 2:
                    next_states_pred = self.forward(states, actions)
                    next_next_states_pred = self.forward(next_states_pred[:, :-1, :], actions[:, 1:, :])
                    # loss = 0.5 * (loss_func(input=next_states_pred, target=next_states) +
                    #               loss_func(input=next_next_states_pred, target=next_states[:, 1:, :]))

                valid_loss += loss.item()
            valid_loss /= len(valid_data_loader)
        return valid_loss
