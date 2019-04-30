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
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_3step_curr_1to3_2.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, bias=True, k_step=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
    model.eval()

    pend_test_data = PendulumDataset('valid')

    prediction_error = np.zeros((len(pend_test_data), pend_test_data.data.shape[1]))
    for j in tqdm(range(len(pend_test_data))):
        states, actions = pend_test_data[j]
        states_sim = states.numpy()
        states_net = torch.zeros(1, 200, 3)
        states_net[0, 0] = states[0]

        for i in range(states.size(0)-1):
            with torch.no_grad():
                encoded, decoded = model.forward(states_net[:, :i+1, :])
                transformed = model.transform(encoded, actions[:i+1])
                transformed_decoded = model.f_decoder(transformed)
                transformed_decoded = transformed_decoded[:, -1, :]

                states_net[0, i+1] = transformed_decoded

        states_net = states_net.view(200, 3).numpy()
        prediction_error[j] = (np.square(states_sim - states_net)).mean(axis=1)

    np.save('./results/checkpoint_16h_1to3step_mse', prediction_error)


if __name__ == '__main__':
    calculate_mse()
