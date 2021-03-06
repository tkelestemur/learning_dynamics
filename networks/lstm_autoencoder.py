import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import utils

class LSTMAutoEncoder(nn.Module):

    def __init__(self, input_size, action_size, hidden_size, num_layers,
                 activation_func='relu', bias=True, k_step=1, lr=1e-3, checkpoint_path=None,
                 encoding='nonlinear', loss_path=None):
        super(LSTMAutoEncoder, self).__init__()

        self.lr = lr
        self.k_step = k_step
        self.checkpoint_path = checkpoint_path
        self.loss_path = loss_path
        self.auto_encoder_type = encoding

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        if encoding == 'linear':
            self.f_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        elif encoding == 'nonlinear':
            self.f_encoder1 = nn.Linear(in_features=input_size, out_features=hidden_size)
            self.f_encoder2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)

            self.f_decoder1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.f_decoder2 = nn.Linear(in_features=hidden_size, out_features=input_size)

        self.f_action = nn.Linear(in_features=action_size, out_features=hidden_size, bias=bias)
        self.f_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)

        self.act_func = getattr(torch, activation_func)

    def forward(self, s):
        if self.auto_encoder_type == 'nonlinear':
            encoded, hidden = self.lstm(self.encode(s))
            decoded = self.decode(encoded)
        elif self.auto_encoder_type == 'linear':
            encoded, hidden = self.lstm(s)
            decoded = self.f_decoder(encoded)
        else:
            raise NotImplementedError

        return encoded, decoded

    def encode(self, state):
        encoded = self.act_func(self.f_encoder1(state))
        encoded = self.f_encoder2(encoded)
        return encoded

    def decode(self, encoded):
        decoded = self.act_func(self.f_decoder1(encoded))
        decoded = self.f_decoder2(decoded)
        return decoded

    def transform(self, s_hidden, a):

        transformed = self.f_hidden(s_hidden) + self.f_action(a)

        return transformed

    def train_model(self, num_epochs=1, train_data_loader=None, valid_data_loader=None,
                    device=None, save_model=False, evaluate=True):

        rec_criteria = nn.MSELoss()
        pred_criteria = nn.MSELoss()
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

        print('Starting training...')
        epoch_loss = np.zeros((num_epochs, 2))
        best_epoch_loss = 100000.0
        for epoch in range(num_epochs):
            self.train()
            train_loss, rec_train_loss, pred_train_loss = 0, 0, 0
            for states, actions in train_data_loader:
                states, actions = states.to(device), actions.to(device)

                self.zero_grad()

                if self.k_step == 1:
                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :])

                elif self.k_step == 2:
                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])

                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = 1/self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                                1/self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :])

                elif self.k_step == 3:
                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])
                    transformed_3 = self.transform(transformed_2[:, :-1, :], actions[:, 2:, :])

                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = 1 / self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                                      1 / self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :]) + \
                                      1 / self.k_step * pred_criteria(input=transformed_3[:, :-1, :], target=encoded[:, 3:, :])


                # encoded, decoded = self.forward(states)
                # encoded_tmp = encoded
                # rec_train_loss = rec_criteria(input=decoded, target=states)
                # pred_weight = 1 / self.k_step
                # for i in range(0, self.k_step):
                #     transformed = self.transform(encoded_tmp, actions[:, i:, :])
                #     pred_train_loss += pred_weight * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, i+1:, :])
                #     encoded_tmp = transformed[:, :-1, :]

                loss = rec_train_loss + pred_train_loss

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

            print('Epoch {} Training Loss: {:.7f} Validation Loss: {:.7f}'.format(epoch + 1, train_loss, valid_loss))
            epoch_loss[epoch] = np.hstack((train_loss, valid_loss))

            is_best = valid_loss <= best_epoch_loss
            if is_best:
                print('New best checkpoint!')
                best_epoch_loss = valid_loss

            if save_model:
                utils.save_checkpoint(self.state_dict(), self.checkpoint_path, is_best)

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
                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :])

                elif self.k_step == 2:
                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])

                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = 1/self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                                1/self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :])

                elif self.k_step == 3:
                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])
                    transformed_3 = self.transform(transformed_2[:, :-1, :], actions[:, 2:, :])

                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = 1 / self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                                      1 / self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :]) + \
                                      1 / self.k_step * pred_criteria(input=transformed_3[:, :-1, :], target=encoded[:, 3:, :])

                loss = rec_train_loss + pred_train_loss
                valid_loss += loss.item()

        valid_loss /= len(valid_data_loader)

        return valid_loss
