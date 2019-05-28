import torch
from tqdm import tqdm
from data import PendulumDataset
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from networks.lstm_autoencoder import LSTMAutoEncoder
from networks import temporal_vae
import utils

plt.style.use('ggplot')


def calculate_mse():
    checkpoint_path = './checkpoints/lstm_auto_encoder/128h_3step_5000_epochs_nonlinear_tanh_best.pth'
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

        with torch.no_grad():
            for t in range(states.size(0) - 1):
                encoded = model.encode(state_t)
                encoded, (h_t, c_t) = model.lstm(encoded, (h_t, c_t))
                transformed = model.transform(encoded, actions[t])
                state_t = model.decode(transformed)
                states_net[t + 1] = state_t.squeeze()

        # with torch.no_grad():
        #     state_hidden_t = model.encode(state_t)
        #     state_hidden_t, (h_t, c_t) = model.lstm(state_hidden_t, (h_t, c_t))
        #     for t in range(states.size(0)-1):
        #         next_state_hiddent_t = model.transform(state_hidden_t, actions[t])
        #         state_t = model.decode(next_state_hiddent_t)
        #         states_net[t+1] = state_t
        #         state_hidden_t = next_state_hiddent_t

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

    torch.save(test_set_pred_error, './results/mse_128h_3step_5000_epochs_nonlinear_tanh_best.pt')


def rollout_temporal_vae():
    config = utils.Config('./configs/config_temporal_vae.yaml')
    checkpoint_path = './checkpoints/temporal_vae/16h_4l_0.0001beta_decoder_best.pth'
    model = temporal_vae.TemporalVAE(input_size=config.input_size,
                                     hidden_size=config.hidden_size,
                                     latent_size=config.latent_size,
                                     k_step=config.k_step).eval()
    utils.load_checkpoint(model, checkpoint_path, 'cpu')

    # trajectory = np.load('pend_data/pendulum_no_action_single_run.npy')
    # trajectory = trajectory[0]

    # pend_valid_data = PendulumDataset('train')
    # trajectory = pend_valid_data[0:50]

    data = np.load('./pend_data/pendulum_no_action_bounded_trajectory.npy')
    trajectory = torch.from_numpy(data[0, :, 0:2])

    sim_init = True

    states_net_dec = torch.zeros(100, 2)
    states_net = torch.zeros(100, 2)

    if sim_init:
        with torch.no_grad():
            mu, logvar = model.encode(trajectory)
            recon = model.decode(mu)
            states_net_dec = recon
            mu_next = model.predict(mu)
            decoded_next = model.decode(mu_next)
            states_net[0] = trajectory[0]
            states_net[1:, :] = decoded_next[:-1, :]
            # states_net = decoded_next
    else:
        with torch.no_grad():

            mu_t, logvar_t = model.encode(trajectory[0])
            decoded_init = model.decode(mu_t)
            states_net[0] = trajectory[0]
            latent_t = model.sample(mu_t, logvar_t)

            for t in range(1, trajectory.size(0)):
                z_next = model.predict(latent_t)
                decoded_next = model.decode(z_next)
                latent_t = z_next
                # z_t = z_next
                states_net[t] = decoded_next
                # print('true z: {}'.format(z_next_true))
                # print('pred z: {}'.format(z_next))

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    ax.set_title('Zero Torque Rollout - beta: 0.0001')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('State')

    ax.plot(trajectory[:, 0].numpy(), label='true state', linewidth=2.5)
    ax.plot(states_net_dec[:, 0].numpy(), '--', label='reconstruction', linewidth=2.5)
    ax.plot(states_net[:, 0].numpy(), '--', label='prediction', linewidth=2.5)

    plt.legend()
    plt.show()


def test_vae():
    config = utils.Config('./configs/config_temporal_vae.yaml')
    checkpoint_path = './checkpoints/temporal_vae/30h_10l_0.0001beta_2step_det.pth'
    model = temporal_vae.TemporalVAE(input_size=config.input_size,
                                     hidden_size=config.hidden_size,
                                     latent_size=config.latent_size,
                                     k_step=config.k_step).eval()
    utils.load_checkpoint(model, checkpoint_path, 'cpu')

    pend_valid_data = PendulumDataset('valid')
    trajectory = pend_valid_data[0]

    with torch.no_grad():
        state_init = trajectory[0:2]
        mu_t, logvar_t = model.encode(state_init)
        decoded_init = model.decode(mu_t)
        # states_net[0] = state_init

        print(trajectory[0:2])
        print(decoded_init)

        z_next = model.f_trainsition(mu_t)
        decoded_next = model.decode(z_next)

        print(trajectory[2:4])
        print(decoded_next)

        z_next = model.f_trainsition(z_next)
        decoded_next = model.decode(z_next)

        print(trajectory[4:6])
        print(decoded_next)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    ax.set_title('Zero Torque Rollout - beta: 0.0001')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('State')

    ax.plot(trajectory[:, 0], label='true state', linewidth=2.5)
    # ax.plot(states_net_dec[:, 0].numpy(), '--', label='reconstruction', linewidth=2.5)
    ax.plot(trajectory[:, 0].numpy(), '--', label='prediction', linewidth=2.5)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # calculate_mse()
    # evaluate_temporal_vae()
    rollout_temporal_vae()
