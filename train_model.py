import torch
from torch.utils.data import DataLoader
from action_conditional_lstm import ActionCondLSTM
from data import PendulumDataset


if __name__ == '__main__':
    # Load dataset
    pend_train_data = PendulumDataset('train')
    pend_test_data = PendulumDataset('test')

    pend_train_loader = DataLoader(dataset=pend_train_data, batch_size=32, drop_last=True,
                                   shuffle=False, num_workers=4)

    pend_valid_loader = DataLoader(dataset=pend_test_data, batch_size=len(pend_test_data),
                                  drop_last=False, shuffle=False, num_workers=2)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_path = './checkpoints/checkpoint_5k_one_steps.pt'
    loss_path = './loss/loss_5k_one_step.csv'

    model = ActionCondLSTM(input_size=3, action_size=1, hidden_size=16, num_layers=1, future_steps=2,
                           checkpoint_path=checkpoint_path, loss_path=loss_path).to(device)

    model.train_model(num_epochs=1000, train_data_loader=pend_train_loader,
                      valid_data_loader=pend_valid_loader, device=device, save_model=True)