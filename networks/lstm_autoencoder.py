import torch
import torch.nn as nn
from torch import optim
import numpy as np


class LSTMAutoEncoder(nn.Module):

    def __init__(self, input_size, action_size, hidden_size, num_layers, future_steps=1, checkpoint_path=None,
                 loss_path=None):
        super(LSTMAutoEncoder, self).__init__()

        self.action_size = action_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.future_steps = future_steps
        self.checkpoint_path = checkpoint_path
        self.loss_path = loss_path

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.f_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.f_action = nn.Linear(in_features=action_size, out_features=hidden_size, bias=False)
        self.f_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, s, a):

        encoded, hidden = self.lstm(s)

        decoded = self.f_decoder(encoded)

        transformed = self.f_hidden(encoded) + self.f_action(a)

        return encoded, decoded, transformed

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
                states, actions = states.to(device), actions.to(device)

                optimizer.zero_grad()

                encoded, decoded, transformed = self.forward(states, actions)
                encdoed_2, decoded_2, transformed_2 = self.forward(decoded[:, :-1, :], actions[:, 1:, :])
                loss = loss_func(input=decoded, target=states) + \
                       loss_func(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                       loss_func(input=transformed_2[:, :-1, :], target=encdoed_2[:, 1:, :])

                loss.backward()
                train_loss += loss.item()

                optimizer.step()

            valid_loss = self.evaluate_model(valid_data_loader=valid_data_loader, device=device)
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
                states, actions = states.to(device), actions.to(device)

                encdoed, decoded, transformed = self.forward(states, actions)

                loss = loss_func(input=decoded, target=states) + \
                       loss_func(input=transformed[:, :-1, :], target=encdoed[:, 1:, :])

                valid_loss += loss.item()
            valid_loss /= len(valid_data_loader)

        return valid_loss
