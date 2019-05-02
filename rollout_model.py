import torch
from tqdm import tqdm
from data import PendulumDataset
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from networks.lstm_autoencoder import LSTMAutoEncoder
plt.style.use('ggplot')


def calculate_mse():
    # Load the model
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_3step.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, k_step=3).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)

    pend_test_data = PendulumDataset('valid')
    prediction_error = np.zeros((len(pend_test_data), pend_test_data.data.shape[1]))

    for j in tqdm(range(len(pend_test_data))):
        states, actions = pend_test_data[j]
        states_sim = states.numpy()
        states_net = torch.zeros(1, 200, 3)
        states_net[0, 0] = states[0]


        state_t = states[0].view(1, 1, 3)
        h_t = torch.zeros(1, 1, 16)
        c_t = torch.zeros(1, 1, 16)
        with torch.no_grad():
            for t in range(states.size(0)-1):
                encoded, (h_t, c_t) = model.lstm(state_t, (h_t, c_t))
                transformed = model.transform(encoded, actions[t])
                state_t = model.f_decoder(transformed)
                states_net[0, t+1] = state_t

        # for i in range(states.size(0)-1):
        #     with torch.no_grad():
        #         encoded, decoded = model.forward(states_net[:, :i+1, :])
        #         transformed = model.transform(encoded, actions[:i+1])
        #         transformed_decoded = model.f_decoder(transformed)
        #         transformed_decoded = transformed_decoded[:, -1, :]
        #
        #         states_net[0, i+1] = transformed_decoded

        # with torch.no_grad():
        #     state_encoded, _ = model.lstm(states[0].view(1, 1, 3))
        # for i in range(states.size(0)-1):
        #     with torch.no_grad():
        #         state_encoded = model.transform(state_encoded, actions[i])
        #         state_decoded = model.f_decoder(state_encoded)
        #         # print(state_decoded.shape)
        #         states_net[0, i+1] = state_decoded[:, -1, :]

        states_net = states_net.view(200, 3).numpy()
        prediction_error[j] = (np.square(states_sim - states_net)).mean(axis=1)

    np.save('./results/checkpoint_16h_3step_single_lstm', prediction_error)

if __name__ == '__main__':
    calculate_mse()
