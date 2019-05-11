import os
import shutil
import torch
import yaml

class Config:

    def __init__(self, config_path=None):
        try:
            with open(config_path, 'r') as f:
                config = yaml.load(f)
        except FileNotFoundError as err:
            print(err)
        except TypeError:
            print('config path is not defined!')

        self.input_size = config['input_size']
        self.action_size = config['action_size']
        self.k_step = config['k_step']
        self.hidden_size = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_layers = config['num_layers']
        self.batch_size = config['batch_size']
        self.bias = config['bias']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.save = config['save']
        self.encoding = config['encoding']
        self.prefix = config['prefix']
        self.curr_learning = config['curr_learning']
        self.pre_trained_path = config['pre_trained_path']

        config_prefix = str(self.hidden_size) + 'h_' + str(self.k_step) + 'step_' + str(self.num_epochs) + '_epochs_' + self.encoding
        self.loss_path = config_prefix + '.csv'
        self.checkpoint_path =  config_prefix + '.pth'


def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def save_checkpoint(state, checkpoint_path, is_best, save_only_best=True):
    if not save_only_best:
        torch.save(state, checkpoint_path)
        if is_best:
            checkpoint_best_path = checkpoint_path.replace('.pth', '_best.pth')
            shutil.copyfile(checkpoint_path, checkpoint_best_path)
    else:
        checkpoint_best_path = checkpoint_path.replace('.pth', '_best.pth')
        torch.save(state, checkpoint_best_path)

def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)

    return checkpoint
