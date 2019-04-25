import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np


class LinearAutoEncoder(nn.Module):

    def __init__(self, input_size, action_size, hidden_size, bias=False, k_step=1, lr=1e-3, checkpoint_path=None,
                 loss_path=None):
        super(LinearAutoEncoder, self).__init__()

        self.lr = lr
        self.k_step = k_step
        self.checkpoint_path = checkpoint_path
        self.loss_path = loss_path

        # self.f_encoder = nn.Sequential(
        #     nn.Linear(in_features=input_size, out_features=hidden_size), nn.ReLU(True),
        #     # nn.Linear(in_features=2*hidden_size, out_features=hidden_size), nn.ReLU(True),
        #     nn.Linear(in_features=hidden_size, out_features=hidden_size)
        #
        # )
        #
        # self.f_decoder = nn.Sequential(
        #     nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.ReLU(True),
        #     # nn.Linear(in_features=hidden_size, out_features=2*hidden_size), nn.ReLU(True),
        #     nn.Linear(in_features=hidden_size, out_features=input_size)
        #
        # )

        self.f_encoder1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.f_encoder2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.f_encoder3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.f_decoder1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.f_decoder2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.f_decoder3 = nn.Linear(in_features=hidden_size, out_features=input_size)

        self.f_action = nn.Linear(in_features=action_size, out_features=hidden_size, bias=bias)
        self.f_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)

    def forward(self, states):
        # encoded = []
        # decoded = []
        # for s_t in s.chunk(s.size(1), dim=1):
        #     encoded_t = self.f_encoder2(F.relu(self.f_encoder1(s_t)))
        #     decoded_t = self.f_decoder2(F.relu(self.f_decoder1(encoded_t)))
        #
        #     encoded += [encoded_t]
        #     decoded += [decoded_t]
        #
        # encoded = torch.stack(encoded, 1)
        # decoded = torch.stack(decoded, 1)

        encoded = self.encode(states)
        decoded = self.decode(encoded)

        return encoded, decoded

    def encode(self, state):

        return self.f_encoder3(F.relu(self.f_encoder2(F.relu(self.f_encoder1(state)))))

    def decode(self, encoded):

        return self.f_decoder3(F.relu(self.f_decoder2(F.relu(self.f_decoder1(encoded)))))

    def transform(self, encoded, a):

        transformed = self.f_hidden(encoded) + self.f_action(a)

        return transformed

    def train_model(self, num_epochs=1, train_data_loader=None, valid_data_loader=None,
                    device=None, save_model=False, evaluate=True):

        rec_criteria = nn.MSELoss()
        pred_criteria = nn.MSELoss()
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

        print('Starting training...')
        epoch_loss = np.zeros((num_epochs, 2))
        for epoch in range(num_epochs):
            self.train()
            train_loss, rec_train_loss, pred_train_loss = 0, 0, 0
            for states, actions in train_data_loader:
                states, actions = states.to(device), actions.to(device)
                # states = states.view(-1, states.size(1)*states.size(2))
                # actions = actions.view(-1, actions.size(1)*actions.size(2))
                # print(states.size())
                self.zero_grad()

                if self.k_step == 1:

                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :])

                # elif self.k_step == 2:
                #     encoded, decoded = self.forward(states)
                #     transformed = self.transform(encoded, actions)
                #     transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])
                #
                #     rec_train_loss = rec_criteria(input=decoded, target=states)
                #     pred_train_loss = 1/self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                #                 1/self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :])
                #
                # elif self.k_step == 3:
                #     encoded, decoded = self.forward(states)
                #     transformed = self.transform(encoded, actions)
                #     transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])
                #     transformed_3 = self.transform(transformed_2[:, :-1, :], actions[:, 2:, :])
                #
                #     rec_train_loss = rec_criteria(input=decoded, target=states)
                #     pred_train_loss = 1 / self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                #                       1 / self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :]) + \
                #                       1 / self.k_step * pred_criteria(input=transformed_3[:, :-1, :], target=encoded[:, 3:, :])

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

            # print('Epoch {} Pred Loss: {:.6f} Rec Loss: {:.6f} Total Loss: {:.6f} Validation Loss: {:.6f}'.format(
            #     epoch + 1, pred_loss, rec_loss, train_loss, valid_loss))

            print('Epoch {} Training Loss: {:.7f} Validation Loss: {:.7f}'.format(epoch + 1, train_loss, valid_loss))

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
                    encoded, decoded = self.forward(states)
                    transformed = self.transform(encoded, actions)
                    rec_train_loss = rec_criteria(input=decoded, target=states)
                    pred_train_loss = pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :])

                # elif self.k_step == 2:
                #     encoded, decoded = self.forward(states)
                #     transformed = self.transform(encoded, actions)
                #     transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])
                #
                #     rec_train_loss = rec_criteria(input=decoded, target=states)
                #     pred_train_loss = 1/self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                #                 1/self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :])
                #
                # elif self.k_step == 3:
                #     encoded, decoded = self.forward(states)
                #     transformed = self.transform(encoded, actions)
                #     transformed_2 = self.transform(transformed[:, :-1, :], actions[:, 1:, :])
                #     transformed_3 = self.transform(transformed_2[:, :-1, :], actions[:, 2:, :])
                #
                #     rec_train_loss = rec_criteria(input=decoded, target=states)
                #     pred_train_loss = 1 / self.k_step * pred_criteria(input=transformed[:, :-1, :], target=encoded[:, 1:, :]) + \
                #                       1 / self.k_step * pred_criteria(input=transformed_2[:, :-1, :], target=encoded[:, 2:, :]) + \
                #                       1 / self.k_step * pred_criteria(input=transformed_3[:, :-1, :], target=encoded[:, 3:, :])

                loss = rec_train_loss + pred_train_loss
                valid_loss += loss.item()

        valid_loss /= len(valid_data_loader)

        return valid_loss
