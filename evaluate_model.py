import torch
from tqdm import tqdm
from data import PendulumDataset
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from networks.lstm_autoencoder import LSTMAutoEncoder
plt.style.use('ggplot')


def calculate_mse():

    checkpoint_path = './checkpoints/lstm_auto_encoder/128h_3step_5000_epochs_best.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=128, num_layers=1, k_step=3).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)

    pend_test_data = PendulumDataset('valid')

    test_set_pred_error = torch.zeros((len(pend_test_data), pend_test_data.data.shape[1]))
    for i in tqdm(range(len(pend_test_data))):

        states, actions = pend_test_data[i]
        states_net = torch.zeros(200, 3)
        states_net[0] = states[0]

        state_t = states[0].view(1, 1, 3)
        h_t = torch.zeros(1, 1, 128)
        c_t = torch.zeros(1, 1, 128)

        # with torch.no_grad():
        #     for t in range(states.size(0)-1):
        #         encoded = model.encode(state_t)
        #         encoded, (h_t, c_t) = model.lstm(encoded, (h_t, c_t))
        #         transformed = model.transform(encoded, actions[t])
        #         state_t = model.decode(transformed)
        #         states_net[t+1] = state_t.squeeze()


        with torch.no_grad():
            state_hidden_t = model.encode(state_t)
            state_hidden_t, (h_t, c_t) = model.lstm(state_hidden_t, (h_t, c_t))
            for t in range(states.size(0)-1):
                next_state_hiddent_t = model.transform(state_hidden_t, actions[t])
                state_t = model.decode(next_state_hiddent_t)
                states_net[t+1] = state_t
                state_hidden_t = next_state_hiddent_t


        # with torch.no_grad():
        #     for t in range(states.size(0)-1):
        #         encoded, (h_t, c_t) = model.lstm(state_t, (h_t, c_t))
        #         transformed = model.transform(encoded, actions[t])
        #         state_t = model.f_decoder(transformed)
        #         states_net[t+1] = state_t.squeeze()

        # with torch.no_grad():
        #     state_encoded, _ = model.lstm(states[0].view(1, 1, 3))
        # for i in range(states.size(0)-1):
        #     with torch.no_grad():
        #         state_encoded = model.transform(state_encoded, actions[i])
        #         state_decoded = model.f_decoder(state_encoded)
        #         # print(state_decoded.shape)
        #         states_net[0, i+1] = state_decoded[:, -1, :]

        error = torch.pow((states - states_net), 2)
        error = torch.mean(error, dim=1)
        test_set_pred_error[i] = error

    torch.save(test_set_pred_error, './results/mse_128h_3step_5000_epochs_best_hidden.pt')

if __name__ == '__main__':
    calculate_mse()
