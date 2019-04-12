import torch
from torch.utils.data import DataLoader
from networks.action_conditional_lstm import ActionCondLSTM
from networks.lstm_autoencoder import LSTMAutoEncoder
from data import PendulumDataset


def train_action_cond_lstm():
    pend_train_data = PendulumDataset('train')
    pend_test_data = PendulumDataset('test')

    pend_train_loader = DataLoader(dataset=pend_train_data, batch_size=32, drop_last=True,
                                   shuffle=False, num_workers=4)

    pend_valid_loader = DataLoader(dataset=pend_test_data, batch_size=len(pend_test_data),
                                   drop_last=False, shuffle=False, num_workers=2)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_path = './checkpoints/checkpoint_5k_one_step_hidden_loss.pt'
    loss_path = './loss/loss_5k_one_step_hidden_loss.csv'

    model = ActionCondLSTM(input_size=3, action_size=1, hidden_size=16, num_layers=1, future_steps=1,
                           checkpoint_path=checkpoint_path, loss_path=loss_path).to(device)

    model.train_model(num_epochs=1000, train_data_loader=pend_train_loader,
                      valid_data_loader=pend_valid_loader, device=device, save_model=False)


def train_lstm_auto_encoder():
    pend_train_data = PendulumDataset('train')
    pend_test_data = PendulumDataset('test')

    pend_train_loader = DataLoader(dataset=pend_train_data, batch_size=16, drop_last=True,
                                   shuffle=False, num_workers=4)

    pend_valid_loader = DataLoader(dataset=pend_test_data, batch_size=len(pend_test_data),
                                   drop_last=False, shuffle=False, num_workers=2)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_5k.pth'
    loss_path = './loss/lstm_auto_encoder/loss_5k.csv'

    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, future_steps=1,
                            checkpoint_path=checkpoint_path, loss_path=loss_path).to(device)

    model.train_model(num_epochs=200, train_data_loader=pend_train_loader,
                      valid_data_loader=pend_valid_loader, device=device, save_model=True)


if __name__ == '__main__':
    # train_action_cond_lstm()
    train_lstm_auto_encoder()
