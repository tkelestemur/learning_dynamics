import torch
from networks import temp_vae
import utils
from data import PendulumDataset


def generate_latent_states():
    config = utils.Config('./configs/config_temporal_vae.yaml')
    checkpoint_path = './checkpoints/temporal_vae/30h_10l_0.0001beta_best.pth'
    model = temp_vae.VAE(input_size=config.input_size,
                         hidden_size=config.hidden_size,
                         latent_size=config.latent_size).eval()
    utils.load_checkpoint(model, checkpoint_path, 'cpu')

    pend_train_data = PendulumDataset('train')