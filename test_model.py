from dm_control import suite
import torch
from torch.utils.data import DataLoader
from data import PendulumDataset
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from networks.lstm_autoencoder import LSTMAutoEncoder
plt.style.use('ggplot')


def test_model():
    # Load the model
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_2step.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, bias=True, k_step=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
    model.eval()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    fig.suptitle('Pendulum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('State')

    # Loadt test set
    pend_test_data = PendulumDataset('test')

    with torch.no_grad():
        # for states, actions in pend_test_data:
        states,  actions = pend_test_data[0]
        trajectory_sim = states.numpy()
        trajectory_net = np.zeros((200, 3))
        trajectory_net[0] = states[0].numpy().copy()
        first_state = states[0].view(1, 1, 3)

        state_hidden, _ = model.lstm(first_state)

        for i in range(actions.size(0)-1):
            state_hidden = model.transform(state_hidden, actions[i])
            state_dec = model.f_decoder(state_hidden)
            trajectory_net[i+1] = state_dec.view(1, 3).numpy()
        #     print('Decoded state: {} True state : {}'.format(state_dec, states[i+1]))

        ax.plot(trajectory_sim[:, 0], c='r', label='true state', linewidth=2)
        ax.plot(trajectory_net[:, 0], '--', c='b', label='prediction w/o bias', linewidth=2)
        plt.legend()
        plt.show()
        # for waypoint in


def run_model():

    # Load the model
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_1step.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, bias=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
    model.eval()

    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_16h_2step.pth'
    model_2 = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1, bias=True)
    model_2.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
    model_2.eval()

    # Plotting
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    fig.suptitle('Pendulum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('State')

    # Environment
    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': 5.0})
    action_spec = env.action_spec()
    env.physics.model.dof_damping[0] = 0.0
    time_step = env.reset()

    state_sim = np.hstack((time_step.observation['orientation'], time_step.observation['velocity']))
    state_net = state_sim.copy()
    state_net_2 = state_sim.copy()
    N = 250
    trajectory_sim = np.zeros((N, 4))
    trajectory_net = np.zeros((N, 4))
    trajectory_net_2 = np.zeros((N, 4))
    trajectory_dec = np.zeros((N, 3))

    i = 0
    while not time_step.last():

        a = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

        trajectory_sim[i] = np.hstack((state_sim, a))
        trajectory_net[i] = np.hstack((state_net, a))
        trajectory_net_2[i] = np.hstack((state_net_2, a))

        trajectory_net_torch = torch.from_numpy(trajectory_net[:i + 1])
        trajectory_net_torch = trajectory_net_torch.view(1, -1, 4).float()

        trajectory_net_2_torch = torch.from_numpy(trajectory_net_2[:i + 1])
        trajectory_net_2_torch = trajectory_net_2_torch.view(1, -1, 4).float()

        with torch.no_grad():
            # model 1
            encoded, decoded, transformed = model.forward(trajectory_net_torch[:, :, 0:3], trajectory_net_torch[:, :, 3:4])
            transformed_decoded = model.f_decoder(transformed)

            # model 2
            encoded_2, decoded_2, transformed_2 = model_2.forward(trajectory_net_2_torch[:, :, 0:3], trajectory_net_2_torch[:, :, 3:4])
            transformed_decoded_2 = model_2.f_decoder(transformed_2)

        # state_dec = decoded.view(decoded.size(1), decoded.size(2))
        # state_dec = state_dec[-1].numpy()
        # trajectory_dec[i] = state_dec

        state_net = transformed_decoded.view(transformed_decoded.size(1), transformed_decoded.size(2))
        state_net = state_net[-1].numpy()

        state_net_2 = transformed_decoded_2.view(transformed_decoded_2.size(1), transformed_decoded_2.size(2))
        state_net_2 = state_net_2[-1].numpy()

        time_step = env.step(a)
        state_sim = np.hstack((time_step.observation['orientation'],
                               time_step.observation['velocity']))
        i += 1

    state_i = 2
    one_step_mse = la.norm(trajectory_sim[:, state_i] - trajectory_net[:, state_i])
    two_step_mse = la.norm(trajectory_sim[:, state_i] - trajectory_net_2[:, state_i])
    ax.plot(trajectory_sim[:, state_i], c='r', label='true state', linewidth=2)
    ax.plot(trajectory_net[:, state_i], '--', c='b', label='prediction w/o bias', linewidth=2)
    ax.plot(trajectory_net_2[:, state_i], '--', c='g', label='prediction w/ bias', linewidth=2)
    print('1-step MSE: {}'.format(one_step_mse))
    print('2-step MSE: {}'.format(two_step_mse))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_model()
    # run_model()
