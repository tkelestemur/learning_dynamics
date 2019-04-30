from dm_control import suite
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data import PendulumDataset
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from networks.lstm_autoencoder import LSTMAutoEncoder
from networks.linear_autoencoder import LinearAutoEncoder
plt.style.use('ggplot')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 8)
fig.suptitle('Pendulum')
ax.set_xlabel('Timestep')
ax.set_ylabel('State')

pend_test_data = PendulumDataset('valid')
states, actions = pend_test_data[457]
states_sim = states.numpy()
ax.plot(states_sim[:, 0], c='r', label='position [true state]', linewidth=2)
ax.plot(states_sim[:, 2], c='b', label='velocity [true state]', linewidth=2)


def trajectory_prediction_lstm():
    # Load the model
    checkpoints = ['checkpoint_16h_1step.pth',
                   'checkpoint_16h_2step.pth',
                   'checkpoint_16h_3step.pth']

    checkpoints_path = './checkpoints/lstm_auto_encoder/'
    # checkpoints = ['checkpoint_16h_1step_lstm_skip.pth']

    # checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_2step.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, bias=True, k_step=1).eval()

    for j, checkpoint in enumerate(checkpoints):
        checkpoint_path = checkpoints_path + checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)

        states_net = torch.zeros(1, 200, 3)
        states_net[0, 0] = states[0]
        for i in range(states.size(0)-1):
            with torch.no_grad():
                encoded, decoded = model.forward(states_net[:, :i+1, :])
                transformed = model.transform(encoded, actions[:i+1])
                transformed_decoded = model.f_decoder(transformed)
                transformed_decoded = transformed_decoded[:, -1, :]

                states_net[0, i+1] = transformed_decoded

        # with torch.no_grad():
        #     state_encoded, _ = model.lstm(states[0].view(1, 1, 3))
        # for i in range(states.size(0)-1):
        #     with torch.no_grad():
        #         state_encoded = model.transform(state_encoded, actions[i])
        #         state_decoded = model.f_decoder(state_encoded)
        #         # print(state_decoded.shape)
        #         states_net[0, i+1] = state_decoded[:, -1, :]

        states_net = states_net.view(200, 3).numpy()
        ax.plot(states_net[:, 0], '--', label='position - ' + checkpoint, linewidth=2)
        ax.plot(states_net[:, 2], '--', label='velocity - ' + checkpoint, linewidth=2)


def trajectory_prediction_linear():
    checkpoints_path = './checkpoints/linear_auto_encoder/'
    checkpoints = ['checkpoint_16h_1step_linear_skip.pth']

    # checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_2step.pth'
    model = LinearAutoEncoder(input_size=3, action_size=1, hidden_size=16, bias=True, k_step=1).eval()

    for j, checkpoint in enumerate(checkpoints):
        checkpoint_path = checkpoints_path + checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
        states_net = torch.zeros(1, 200, 3)
        states_net[0, 0] = states[0]
        for i in range(states.size(0)-1):
            with torch.no_grad():
                encoded, decoded = model.forward(states_net[:, :i+1, :])
                transformed = model.transform(encoded, actions[:i+1])
                transformed_decoded = model.decode(transformed)
                transformed_decoded = transformed_decoded[:, -1, :]

                states_net[0, i+1] = transformed_decoded

        # with torch.no_grad():
        #     state_encoded, _ = model.lstm(states[0].view(1, 1, 3))
        # for i in range(states.size(0)-1):
        #     with torch.no_grad():
        #         state_encoded = model.transform(state_encoded, actions[i])
        #         state_decoded = model.f_decoder(state_encoded)
        #         # print(state_decoded.shape)
        #         states_net[0, i+1] = state_decoded[:, -1, :]

        states_net = states_net.view(200, 3).numpy()
        ax.plot(states_net[:, 0], '--', label='position - ' + checkpoint, linewidth=2)
        ax.plot(states_net[:, 2], '--', label='velocity - ' + checkpoint, linewidth=2)


def calculate_mse():
    # Load the model
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_1step.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, bias=True, k_step=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
    model.eval()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    fig.suptitle('Pendulum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Prediction MSE')

    pend_test_data = PendulumDataset('valid')

    num_samples = 1
    prediction_error = np.zeros((num_samples, pend_test_data.data.shape[1]))
    for j in tqdm(range(num_samples)):
        states, actions = pend_test_data[j]
        states_sim = states.cpu().numpy()
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

    print(states_net[2])
    print(states_sim[2])
    print(prediction_error[0, 2])
    # prediction_error_mean = prediction_error.mean(axis=0)
    # np.save('./results/checkpoint_16h_1step_mse', prediction_error)
    # ax.plot(prediction_error_mean, c='r', label='prediction error', linewidth=2)
    # plt.show()


if __name__ == '__main__':
    trajectory_prediction_lstm()
    # trajectory_prediction_linear()
    plt.legend()
    plt.show()

    # calculate_mse()
