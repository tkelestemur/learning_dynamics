import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import utils


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, activation_func='tanh'):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder
        self.f_enc1 = nn.Linear(input_size, hidden_size)
        self.f_mu = nn.Linear(hidden_size, latent_size)
        self.f_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.f_dec1 = nn.Linear(latent_size, hidden_size)
        self.f_dec2 = nn.Linear(hidden_size, input_size)

        # Activation Function
        self.act_func = getattr(torch, activation_func)

    def encode(self, state):
        hidden = self.act_func(self.f_enc1(state))
        return self.f_mu(hidden), self.f_logvar(hidden)

    def decode(self, hidden):
        hidden = self.act_func(self.f_dec1(hidden))
        decoded = self.f_dec2(hidden)
        return decoded

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, states):

        # Reconstruct the trajectory
        mu, logvar = self.encode(states)
        latent = self.sample(mu, logvar)
        recon = self.decode(latent)

        return recon, mu, logvar

    def loss_func(self, recon, input, mu, logvar):
        MSE_REC = F.mse_loss(input=recon, target=input, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / input.size(1)

        return MSE_REC, KLD


class Transition(nn.Module):
    def __init__(self, latent_size, k_step):
        super(Transition, self).__init__()

        # Transition Function
        self.f_trainsition = nn.Linear(latent_size, latent_size)
        self.k_step = k_step

    def forward(self, latent):
        latent_next = self.f_trainsition(latent[:, :-1, :])

        return latent_next, latent[:, 1:, :]

    def loss_func(self, latent_next, latent):
        MSE_PRED = F.mse_loss(input=latent_next, target=latent)
        return MSE_PRED


def train_epoch(model, optimizer, beta, train_data_loader, device, model_type):
    model.train()
    total_loss, recon_loss, mse_t_loss, kld_loss = 0, 0, 0, 0
    for states in train_data_loader:
        states_batch = states.to(device)

        optimizer.zero_grad()
        if model_type == 'vae':
            recon, mu, logvar = model(states_batch)
            MSE_REC, KLD = model.loss_func(recon, states_batch, mu, logvar)
            vae_loss = MSE_REC + beta * KLD
            vae_loss.backward()

            total_loss += vae_loss.item()
            recon_loss += MSE_REC.item()
            kld_loss += KLD.item()
        elif model_type == 'transition':
            latent_next, latent = model(states_batch)
            pred_loss = model.loss_func(latent_next, latent)
            pred_loss.backward()
            mse_t_loss = pred_loss.item()

        optimizer.step()

    total_loss /= len(train_data_loader.dataset)
    recon_loss /= len(train_data_loader.dataset)
    mse_t_loss /= len(train_data_loader.dataset)
    kld_loss /= len(train_data_loader.dataset)

    loss_dict = {'total': total_loss, 'mse': recon_loss,
                 'mse_t': mse_t_loss, 'kld': kld_loss}

    return loss_dict


def evaluate_epoch(model, beta, valid_data_loader, device, model_type):
    model.eval()
    total_loss, recon_loss, mse_t_loss, kld_loss = 0, 0, 0, 0
    for states in valid_data_loader:
        states_batch = states.to(device)
        if model_type == 'vae':
            recon, mu, logvar = model(states_batch)
            MSE_REC, KLD = model.loss_func(recon, states_batch, mu, logvar)
            vae_loss = MSE_REC + beta * KLD

            total_loss += vae_loss.item()
            recon_loss += MSE_REC.item()
            kld_loss += KLD.item()
        elif model_type == 'transition':
            latent_next, latent = model(states_batch)
            pred_loss = model.loss_func(latent_next, latent)
            mse_t_loss = pred_loss.item()

    total_loss /= len(valid_data_loader.dataset)
    recon_loss /= len(valid_data_loader.dataset)
    mse_t_loss /= len(valid_data_loader.dataset)
    kld_loss /= len(valid_data_loader.dataset)

    loss_dict = {'total': total_loss, 'mse': recon_loss,
                 'mse_t': mse_t_loss, 'kld': kld_loss}

    return loss_dict


def train(model, config, train_data_loader, valid_data_loader, checkpoint_path, loss_path, device, model_type):
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    print('Starting training...')
    best_eval_loss = 100000.0
    # betas = np.linspace(0, 0.1, config.num_epochs)
    epoch_loss = np.zeros((config.num_epochs, 8))

    for epoch_i in range(config.num_epochs):
        train_loss_dict = train_epoch(model, optimizer, config.beta, train_data_loader, device, model_type)
        valid_loss_dict = evaluate_epoch(model, config.beta, valid_data_loader, device, model_type)

        print('====> Epoch: {}/{}'.format(epoch_i+1, config.num_epochs))
        print('====> Train loss:     {:.7f} Eval Loss: {:.7f} '.format(train_loss_dict['total'],
                                                                       valid_loss_dict['total']))

        print('====> Train MSE Loss: {:.7f} MSE T Loss {:.7f} KLD Loss:  {:.7f} '.format(train_loss_dict['mse'],
                                                                                         train_loss_dict['mse_t'],
                                                                                         train_loss_dict['kld']))

        print('====> Eval  MSE Loss: {:.7f} MSE T Loss {:.7f} KLD Loss:  {:.7f} '.format(valid_loss_dict['mse'],
                                                                                           valid_loss_dict['mse_t'],
                                                                                           valid_loss_dict['kld']))

        epoch_loss[epoch_i] = np.hstack((list(train_loss_dict.values()), list(valid_loss_dict.values())))

        is_best = valid_loss_dict['total'] <= best_eval_loss
        if is_best:
            print('====> New best checkpoint!')
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
