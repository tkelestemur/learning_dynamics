import torch
import torch.nn as nn
from torch import optim
import numpy as np


class LSTMAutoEncoder(nn.Module):

    def __init__(self, input_size, action_size, hidden_size, num_layers, bias=False, k_step=1, checkpoint_path=None,
                 loss_path=None):
        super(LSTMAutoEncoder, self).__init__()

        self.action_size = action_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k_step = k_step
        self.checkpoint_path = checkpoint_path
        self.loss_path = loss_path

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.f_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.f_action = nn.Linear(in_features=action_size, out_features=hidden_size, bias=bias)
        self.f_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)

    def forward(self, s, a):

        encoded, hidden = self.lstm(s)

        decoded = self.f_decoder(encoded)

        transformed = self.transform(encoded, a)

        return encoded, decoded, transformed

    def transform(self, s_hidden, a):

        transformed = self.f_hidden(s_hidden) + self.f_action(a)

        return transformed

    def train_model(self, num_epochs=1, train_data_loader=None, valid_data_loader=None,
                    device=None, save_model=False, evaluate=True):

        rec_criteria = nn.MSELoss()
        pred_criteria = nn.MSELoss()
        optimizer = optim.Adam(params=self.parameters(), lr=1e-3)

        print('Starting training...\nNumber of future step predictions: {}'.format(self.k_step))
        epoch_loss = np.zeros((num_epochs, 2))
        for epoch in range(num_epochs):
            self.train()
            train_loss, rec_loss, pred_loss = 0, 0, 0
            for states, actions in train_data_loader:
                states, actions = states.to(device), actions.to(device)

                self.zero_grad()

                if self.k_step == 1:
                    encoded, decoded, transformed = self.forward(states, actions)
                    rec_loss = rec_criteria(input=decoded, target=states)
                    pred_loss = pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :])

                elif self.k_step == 2:
                    encoded, decoded, transformed = self.forward(states, actions)
                    transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])

                    rec_loss = rec_criteria(input=decoded, target=states)
                    pred_loss = 1/self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                                1/self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :])
                else:
                    raise NotImplementedError

                loss = rec_loss + pred_loss

                # rec_loss += rec_loss.item()
                # pred_loss += pred_loss.item()
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            train_loss = train_loss / len(train_data_loader)
            # rec_loss = rec_loss / len(train_data_loader)
            # pred_loss = pred_loss / len(train_data_loader)

            if evaluate:
                valid_loss = self.evaluate_model(valid_data_loader=valid_data_loader, device=device)
            else:
                valid_loss = 0

            # print('Epoch {} Pred Loss: {:.6f} Rec Loss: {:.6f} Total Loss: {:.6f} Validation Loss: {:.6f}'.format(
            #     epoch + 1, pred_loss, rec_loss, train_loss, valid_loss))

            print('Epoch {} Training Loss: {:.6f} Validation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

            if save_model:
                epoch_loss[epoch] = np.hstack((train_loss, valid_loss))
                if not self.checkpoint_path:
                    print('Checkpoint path is not specified! Can\'t save the model..')
                else:
                    with open(self.checkpoint_path, 'wb') as f:
                        torch.save(self.state_dict(), f)

        if save_model:
            with open(self.loss_path, 'wb') as f:
                np.savetxt(f, epoch_loss, delimiter=',')

    def evaluate_model(self, valid_data_loader=None, device=None):
        self.eval()
        rec_criteria = nn.MSELoss()
        pred_criteria = nn.MSELoss()

        valid_loss = 0
        with torch.no_grad():
            for states, actions in valid_data_loader:
                states, actions = states.to(device), actions.to(device)

                if self.k_step == 1:
                    encoded, decoded, transformed = self.forward(states, actions)
                    rec_loss = rec_criteria(input=decoded, target=states)
                    pred_loss = pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :])

                elif self.k_step == 2:
                    encoded, decoded, transformed = self.forward(states, actions)
                    # encdoed_2, decoded_2, transformed_2 = self.forward(decoded[:, :-1, :], actions[:, 1:, :])
                    transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])

                    rec_loss = rec_criteria(input=decoded, target=states)
                    pred_loss = 1 / self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                                1 / self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :])
                else:
                    raise NotImplementedError

                loss = rec_loss + pred_loss
                valid_loss += loss.item()

        valid_loss /= len(valid_data_loader)

        return valid_loss
