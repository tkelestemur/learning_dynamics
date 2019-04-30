import yaml
import torch
from torch.utils.data import DataLoader
from networks.action_conditional_lstm import ActionCondLSTM
from networks.lstm_autoencoder import LSTMAutoEncoder
from networks.linear_autoencoder import LinearAutoEncoder
from data import PendulumDataset

pend_train_data = PendulumDataset('train')
pend_valid_data = PendulumDataset('valid')

pend_train_loader = DataLoader(dataset=pend_train_data, batch_size=32,
                               drop_last=True, shuffle=False, num_workers=4)

pend_valid_loader = DataLoader(dataset=pend_valid_data, batch_size=len(pend_valid_data),
                               drop_last=False, shuffle=False, num_workers=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_action_cond_lstm():
    checkpoint_path = './checkpoints/checkpoint_5k_one_step_hidden_loss.pt'
    loss_path = './loss/loss_5k_one_step_hidden_loss.csv'

    model = ActionCondLSTM(input_size=3, action_size=1, hidden_size=16, num_layers=1, future_steps=1,
                           checkpoint_path=checkpoint_path, loss_path=loss_path).to(device)

    model.train_model(num_epochs=1000, train_data_loader=pend_train_loader,
                      valid_data_loader=pend_valid_loader, device=device, save_model=False)


def train_lstm_auto_encoder(config):
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_' + str(config['hidden_size']) + 'h_' + str(
        config['k_step']) + 'step' + config['prefix'] + '.pth'
    loss_path = './loss/lstm_auto_encoder/loss_' + str(config['hidden_size']) + 'h_' + str(config['k_step']) + 'step' + \
                config['prefix'] + '.csv'

    model = LSTMAutoEncoder(input_size=config['input_size'], action_size=config['action_size'], lr=config['lr'],
                            hidden_size=config['hidden_size'], num_layers=config['num_layers'], bias=config['bias'],
                            k_step=config['k_step'], checkpoint_path=checkpoint_path, loss_path=loss_path).to(device)

    if config['curr_learning']:
        model.load_state_dict(torch.load(config['pre_trained_path'], map_location=device), strict=True)

    model.train_model(num_epochs=config['num_epochs'], train_data_loader=pend_train_loader,
                      valid_data_loader=pend_valid_loader, device=device, save_model=config['save'])


def train_linear_auto_encoder(config):
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_' + str(config['hidden_size']) + 'h_' + str(
        config['k_step']) + 'step' + config['prefix'] + '.pth'
    loss_path = './loss/lstm_auto_encoder/loss_' + str(config['hidden_size']) + 'h_' + str(
        config['k_step']) + 'step' + config['prefix'] + '.csv'

    model = LinearAutoEncoder(input_size=config['input_size'], action_size=config['action_size'], lr=config['lr'],
                              hidden_size=config['hidden_size'], bias=config['bias'], k_step=config['k_step'],
                              checkpoint_path=checkpoint_path, loss_path=loss_path).to(device)

    model.train_model(num_epochs=config['num_epochs'], train_data_loader=pend_train_loader,
                      valid_data_loader=pend_valid_loader, device=device, save_model=config['save'])


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    print('Training parameters: {}'.format(config['parameters']))
    train_lstm_auto_encoder(config['parameters'])
    # train_linear_auto_encoder(config['parameters'])
    # train_action_cond_lstm()
