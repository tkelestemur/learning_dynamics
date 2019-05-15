import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import utils


class TemporalVAE(nn.Module):

    def __init__(self, input_size, hidden_size, latent_size, activation_func='tanh'):
        super(TemporalVAE, self).__init__()

        # Encoder
        self.f_enc1 = nn.Linear(input_size, hidden_size)
        self.f_mu = nn.Linear(hidden_size, latent_size)
        self.f_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.f_dec1 = nn.Linear(latent_size, hidden_size)
        self.f_dec2 = nn.Linear(hidden_size, input_size)

        # Transition Function
        self.f_trainsition = nn.Linear(latent_size, latent_size)

        # Activation Function
        self.act_func = getattr(torch, activation_func)

    def encode(self, state):
        hidden = self.act_func(self.f_enc1(state))
        return self.f_mu(hidden), self.f_logvar(hidden)

    def decode(self, hidden):
        hidden = self.act_func(self.f_dec1(hidden))
        decoded = self.f_dec2(hidden)
        # decoded = torch.sigmoid(self.f_dec2(hidden))
        return decoded

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, state):
        mu, logvar = self.encode(state[:, 0:2])
        z = self.sample(mu, logvar)

        mu_next, logvar_next = self.encode(state[:, 2:4])
        z_next = self.sample(mu_next, logvar_next)

        z_next_hat = self.f_trainsition(z)
        return self.decode(z), mu, logvar, z_next, z_next_hat


mse_loss = nn.MSELoss()
def loss_func(decoded, state, mu, logvar, z_next, z_next_hat, beta=1):
    MSE = mse_loss(input=decoded, target=state)
    MSE_T = mse_loss(input=z_next_hat, target=z_next)
    KLD = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, MSE_T, KLD

def train_epoch(model, optimizer, beta, train_data_loader, checkpoint_path, device):
    model.train()
    train_loss, mse_loss, mse_t_loss, kld_loss = 0, 0, 0, 0
    for states in train_data_loader:
        states_batch = states.to(device)

        optimizer.zero_grad()
        decoded_batch, mu, logvar, z_next, z_next_hat = model(states_batch)
        MSE, MSE_T, KLD = loss_func(decoded_batch, states_batch[:, 0:2], mu, logvar, z_next, z_next_hat, beta=beta)
        loss = MSE + MSE_T + KLD

        loss.backward()

        train_loss += loss.item()
        mse_loss += MSE.item()
        mse_t_loss += MSE_T.item()
        kld_loss += KLD.item()

        optimizer.step()

    train_loss = train_loss / len(train_data_loader)
    mse_loss = mse_loss / len(train_data_loader)
    mse_t_loss = mse_t_loss / len(train_data_loader)
    kld_loss = kld_loss / len(train_data_loader)

    loss_dict = {'total': train_loss, 'mse': mse_loss,
                 'mse_t': mse_t_loss, 'kld': kld_loss}

    return loss_dict


def evaluate_epoch(model, beta, valid_data_loader, device):
    model.eval()
    eval_loss, mse_eval_loss, mse_t_eval_loss, kld_eval_loss = 0, 0, 0, 0
    with torch.no_grad():
        for states in valid_data_loader:
            states_batch = states.to(device)

            decoded_batch, mu, logvar, z_next, z_next_hat = model(states_batch)
            MSE, MSE_T, KLD = loss_func(decoded_batch, states_batch[:, 0:2], mu, logvar, z_next, z_next_hat, beta=beta)

            loss = MSE + MSE_T + KLD

            mse_eval_loss += MSE.item()
            mse_t_eval_loss += MSE_T.item()
            kld_eval_loss += KLD.item()
            eval_loss += loss.item()

        eval_loss /= len(valid_data_loader)
        mse_eval_loss /= len(valid_data_loader)
        mse_t_eval_loss /= len(valid_data_loader)
        kld_eval_loss /= len(valid_data_loader)

        loss_dict = {'total': eval_loss, 'mse': mse_eval_loss,
                     'mse_t': mse_t_eval_loss, 'kld': kld_eval_loss}

    return loss_dict


def train(model, config, train_data_loader, valid_data_loader, checkpoint_path, loss_path, device):

    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    print('Starting training...')
    best_eval_loss = 100000.0
    # betas = np.linspace(0, 0.1, config.num_epochs)
    epoch_loss = np.zeros((config.num_epochs, 8))

    for epoch_i in range(config.num_epochs):
        train_loss_dict = train_epoch(model, optimizer, config.beta, train_data_loader, checkpoint_path, device)
        valid_loss_dict = evaluate_epoch(model, config.beta, valid_data_loader, device)

        print('====> Epoch: {} Train loss: {} Eval Loss: {} '.format(epoch_i+1, train_loss_dict['total'], valid_loss_dict['total']))
        print('====> Train MSE Loss:       {} MSE T Loss {} KLD Loss:  {} '.format(train_loss_dict['mse'], train_loss_dict['mse_t'], train_loss_dict['kld']))
        print('====> Eval  MSE Loss:       {} MSE T Loss {} KLD Loss:  {} '.format(valid_loss_dict['mse'], valid_loss_dict['mse_t'], valid_loss_dict['kld']))

        epoch_loss[epoch_i] = np.hstack((list(train_loss_dict.values()), list(valid_loss_dict.values())))

        is_best = valid_loss_dict['total'] <= best_eval_loss
        if is_best:
            print('New best checkpoint!')
            best_eval_loss = valid_loss_dict['total']

        if config.save:
            utils.save_checkpoint(state=model.state_dict(),
                                  checkpoint_path=checkpoint_path,
                                  is_best=is_best,
                                  save_only_best=False)

        print('----------------------------------------------------------------------------')

    if config.save:
        with open(loss_path, 'wb') as f:
            np.savetxt(f, epoch_loss, delimiter=',')
