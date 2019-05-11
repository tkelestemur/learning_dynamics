import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import utils


class TemporalVAE(nn.Module):

    def __init__(self, input_size, hidden_size, latent_size):
        super(TemporalVAE, self).__init__()

        self.f_enc1 = nn.Linear(input_size, hidden_size)
        self.f_mu = nn.Linear(hidden_size, latent_size)
        self.f_logvar = nn.Linear(hidden_size, latent_size)

        self.f_dec1 = nn.Linear(latent_size, hidden_size)
        self.f_dec2 = nn.Linear(hidden_size, input_size)

    def encode(self, state):
        hidden = F.relu(self.f_enc1(state))
        return self.f_mu(hidden), self.f_logvar(hidden)

    def decode(self, hidden):
        hidden = F.relu(self.f_dec1(hidden))
        return torch.sigmoid(self.f_dec2(hidden))

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, state):
        mu, logvar = self.encode(state)
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_func(self, decoded, state, mu, logvar):
        BCE = F.binary_cross_entropy(decoded, state)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def train_epoch(model, optimizer, train_data_loader, device):
    model.train()
    train_loss = 0
    for states in train_data_loader:
        states_batch = states.to(device)

        decoded_batch, mu, logvar = model(states_batch[:, 0:2])
        loss = model.loss_func(decoded_batch, states_batch[:, 0:2], mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss = train_loss / len(train_data_loader)
    return train_loss


def evaluate_epoch(model, valid_data_loader, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for states in valid_data_loader:
            states_batch = states.to(device)
            decoded_batch, mu, logvar = model(states_batch[:, 0:2])
            loss = model.loss_func(decoded_batch, states_batch[:, 0:2], mu, logvar)
            eval_loss += loss.item()

        eval_loss = eval_loss / len(valid_data_loader)

    return eval_loss


def train(model, config, train_data_loader, valid_data_loader):

    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    device = utils.get_device()
    print('Starting training...')
    for epoch_i in range(config.num_epochs):
        train_loss = train_epoch(model, optimizer, train_data_loader, device)
        eval_loss = evaluate_epoch(model, valid_data_loader, device)

        print('====> Epoch: {} Train loss: {:.4f} Eval Loss: {:.4f}'.format(epoch_i, train_loss, eval_loss))
