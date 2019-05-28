import torch
from torch.utils.data import DataLoader
from networks import temp_vae
from data import PendulumDataset
from utils import Config
import utils


def train_vae():

    config = Config('./configs/config_temporal_vae.yaml')
    checkpoint_path = './checkpoints/temporal_vae/' + config.checkpoint_path
    loss_path = './loss/temporal_vae/' + config.loss_path
    device = utils.get_device()

    pend_train_data = PendulumDataset('train')
    pend_valid_data = PendulumDataset('valid')

    pend_train_loader = DataLoader(dataset=pend_train_data,
                                   batch_size=config.batch_size,
                                   drop_last=True,
                                   shuffle=False,
                                   num_workers=4)

    pend_valid_loader = DataLoader(dataset=pend_valid_data,
                                   batch_size=len(pend_valid_data),
                                   drop_last=False,
                                   shuffle=False,
                                   num_workers=2)

    model = temp_vae.VAE(input_size=config.input_size,
                         hidden_size=config.hidden_size,
                         latent_size=config.latent_size).to(device)

    temp_vae.train(model=model,
                   config=config,
                   train_data_loader=pend_train_loader,
                   valid_data_loader=pend_valid_loader,
                   checkpoint_path=checkpoint_path,
                   loss_path=loss_path,
                   device=device,
                   model_type=config.model_type)


def train_transition():
    config = Config('./configs/config_temporal_vae.yaml')
    checkpoint_path = './checkpoints/temporal_vae/' + config.checkpoint_path
    loss_path = './loss/temporal_vae/' + config.loss_path
    device = utils.get_device()

    pend_train_data = PendulumDataset('train')
    pend_valid_data = PendulumDataset('valid')

    pend_train_loader = DataLoader(dataset=pend_train_data,
                                   batch_size=config.batch_size,
                                   drop_last=True,
                                   shuffle=False,
                                   num_workers=4)

    pend_valid_loader = DataLoader(dataset=pend_valid_data,
                                   batch_size=len(pend_valid_data),
                                   drop_last=False,
                                   shuffle=False,
                                   num_workers=2)

    model = temp_vae.Transition(latent_size=config.latent_size).to(device)

    temp_vae.train(model=model,
                   config=config,
                   train_data_loader=pend_train_loader,
                   valid_data_loader=pend_valid_loader,
                   checkpoint_path=checkpoint_path,
                   loss_path=loss_path,
                   device=device,
                   model_type=config.model_type)


if __name__ == '__main__':
    train_vae()
