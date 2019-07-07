import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import utils


class TemporalVAE(nn.Module):

    def __init__(self, input_size, hidden_size, latent_size, k_step, activation_func='tanh'):
        super(TemporalVAE, self).__init__()

        self.k_step = k_step
        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder
        self.f_enc1 = nn.Linear(input_size, hidden_size)
        self.f_mu = nn.Linear(hidden_size, latent_size)
        self.f_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.f_dec1 = nn.Linear(latent_size, hidden_size)
        self.f_dec2 = nn.Linear(hidden_size, input_size)

        # Transition Function
        self.f_trainsition = nn.Linear(latent_size, latent_size)
        # self.f_trainsition2 = nn.Linear(latent_size, latent_size)

        # Activation Function
        self.act_func = getattr(torch, activation_func)

    def encode(self, state):
        hidden = self.act_func(self.f_enc1(state))
        return self.f_mu(hidden), self.f_logvar(hidden)

    def predict(self, prev_states):
        next_states = self.f_trainsition(prev_states)
        # next_state = self.f_trainsition2(hidden)
        return next_states

    def decode(self, hidden):
        hidden = self.act_func(self.f_dec1(hidden))
        decoded = self.f_dec2(hidden)
        # decoded = torch.sigmoid(decoded)
        return decoded

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, states):

        # Reconstruct the trajectory
        mu, logvar = self.encode(states[:, 0:2])
        latent = self.sample(mu, logvar)
        recon = self.decode(latent)

        z_next_pred = self.predict(latent)

        mu_next, logvar_next = self.encode(states[:, 3:5])
        z_next_sampled = self.sample(mu_next, logvar_next)

        return recon, mu, logvar, z_next_sampled, z_next_pred

    def loss_func(self, recon, states, mu, logvar, z_next_sampled, z_next_pred):
        MSE_REC = F.mse_loss(input=recon, target=states[:, 0:2], reduction='sum')
        MSE_PRED = F.mse_loss(input=z_next_pred, target=z_next_sampled, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE_REC, MSE_PRED, KLD


def train_epoch(model, optimizer, beta, train_data_loader, device):
    model.train()
    total_loss, recon_loss, pred_loss, kld_loss = 0, 0, 0, 0
    for states in train_data_loader:
        states_batch = states.to(device)

        optimizer.zero_grad()

        recon, mu, logvar, z_next_sampled, z_next_pred = model(states_batch)
        MSE_REC, MSE_PRED, KLD = model.loss_func(recon, states_batch, mu, logvar, z_next_sampled, z_next_pred)
        # loss = MSE_REC + beta * KLD + MSE_PRED
        loss = MSE_REC + MSE_PRED
        loss.backward()
        total_loss += loss.item()
        recon_loss += MSE_REC.item()
        pred_loss += MSE_PRED.item()
        kld_loss += KLD.item()

        optimizer.step()

    total_loss /= len(train_data_loader.dataset)
    recon_loss /= len(train_data_loader.dataset)
    pred_loss /= len(train_data_loader.dataset)
    kld_loss /= len(train_data_loader.dataset)

    loss_dict = {'total': total_loss, 'recon': recon_loss,
                 'pred': pred_loss, 'kld': kld_loss}

    return loss_dict


def evaluate_epoch(model, beta, valid_data_loader, device):
    model.eval()
    total_loss, recon_loss, pred_loss, kld_loss = 0, 0, 0, 0
    with torch.no_grad():
        for states in valid_data_loader:
            states_batch = states.to(device)

            recon, mu, logvar, z_next_sampled, z_next_pred = model(states_batch)
            MSE_REC, MSE_PRED, KLD = model.loss_func(recon, states_batch, mu, logvar, z_next_sampled, z_next_pred)
            # loss = MSE_REC + MSE_PRED + beta * KLD
            loss = MSE_REC + MSE_PRED

            total_loss += loss.item()
            recon_loss += MSE_REC.item()
            pred_loss += MSE_PRED.item()
            kld_loss += KLD.item()

        total_loss /= len(valid_data_loader.dataset)
        recon_loss /= len(valid_data_loader.dataset)
        pred_loss /= len(valid_data_loader.dataset)
        kld_loss /= len(valid_data_loader.dataset)

        loss_dict = {'total': total_loss, 'recon': recon_loss,
                     'pred': pred_loss, 'kld': kld_loss}

    return loss_dict


def train(model, config, train_data_loader, valid_data_loader, checkpoint_path, loss_path, device):
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    print('Starting training...')
    best_eval_loss = 100000.0
    # betas = np.linspace(0, 0.1, config.num_epochs)
    epoch_loss = np.zeros((config.num_epochs, 8))

    for epoch_i in range(config.num_epochs):
        train_loss_dict = train_epoch(model, optimizer, config.beta, train_data_loader, device)
        valid_loss_dict = evaluate_epoch(model, config.beta, valid_data_loader, device)

        print('====> Epoch: {}/{}'.format(epoch_i+1, config.num_epochs))
        print('====> Train loss:     {:.7f} Eval Loss: {:.7f} '.format(train_loss_dict['total'],
                                                                       valid_loss_dict['total']))

        print('====> Train MSE Loss: {:.7f} MSE T Loss {:.7f} KLD Loss:  {:.7f} '.format(train_loss_dict['recon'],
                                                                                         train_loss_dict['pred'],
                                                                                         train_loss_dict['kld']))

        print('====> Eval  MSE Loss: {:.7f} MSE T Loss {:.7f} KLD Loss:  {:.7f} '.format(valid_loss_dict['recon'],
                                                                                         valid_loss_dict['pred'],
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
