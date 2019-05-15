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
            for t in range(states.size(0)-1):
                encoded = model.encode(state_t)
                encoded, (h_t, c_t) = model.lstm(encoded, (h_t, c_t))
                transformed = model.transform(encoded, actions[t])
                state_t = model.decode(transformed)
                states_net[t+1] = state_t.squeeze()


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
        from dm_control import suite
        config = utils.Config('./configs/config_temporal_vae.yaml')
        # pend_train_data = PendulumDataset('train')
        # pend_valid_data = PendulumDataset('valid')
        checkpoint_path = './checkpoints/temporal_vae/30h_10l_0.001beta_best.pth'

        model = temporal_vae.TemporalVAE(input_size=config.input_size,
                                         hidden_size=config.hidden_size,
                                         latent_size=config.latent_size).eval()
        utils.load_checkpoint(model, checkpoint_path, 'cpu')

        # env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': 2.0})
        trajectory = np.load('pend_data/pendulum_no_action_single_run.npy')
        trajectory = trajectory[0]
        # time_step = env.reset()
        # with torch.no_grad():
        #     while not time_step.last():
        #         state = time_step.observation['orientation']
        #         state = state.astype(np.float32)
        #         state = torch.from_numpy(state)
        #         if time_step.first():
        #             mu, logvar = model.encode(state)
        #             decoded = model.decode(mu)
        #         else:
        #             z_next = model.f_trainsition(mu)
        #             decoded = model.decode(z_next)
        #             mu = z_next
        #         print('true state:    {}'.format(state))
        #         print('decoded state: {}'.format(decoded))
        #         time_step = env.step(0.0)

        states_net_dec = torch.zeros(100, 2)
        states_net = torch.zeros(101, 2)
        states_true = np.zeros((100, 2))

        # t = 0
        with torch.no_grad():
            # state = time_step.observation['orientation']
            for t in range(trajectory.shape[0]):
                # states_true[t] = time_step.observation['orientation']
                states_true[t] = trajectory[t]
                state = states_true[t].astype(np.float32)
                state = torch.from_numpy(state)

                mu, logvar = model.encode(state)
                decoded = model.decode(mu)
                z_next = model.f_trainsition(mu)
                decoded_next = model.decode(z_next)

                if t == 0:
                    states_net[t] = state
                    states_net[t+1] = decoded_next
                else:
                    states_net[t+1] = decoded_next
                states_net_dec[t] = decoded

                # time_step = env.step(0.0)
                # t += 1

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 8)
        ax.set_title('Zero Torque Rollout - beta: 0.001')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('State')

        ax.plot(states_true[:, 0], label='true state', linewidth=2.5)
        ax.plot(states_net_dec[:, 0].numpy(), '--', label='reconstruction', linewidth=2.5)
        ax.plot(states_net[:100, 0].numpy(), '--', label='prediction', linewidth=2.5)

        plt.legend()
        plt.show()

def evaluate_temporal_vae():

    config = utils.Config('./configs/config_temporal_vae.yaml')
    # pend_train_data = PendulumDataset('train')
    pend_valid_data = PendulumDataset('valid')
    checkpoint_path = './checkpoints/temporal_vae/30h_10l_0.001beta_best.pth'

    model = temporal_vae.TemporalVAE(input_size=config.input_size,
                                     hidden_size=config.hidden_size,
                                     latent_size=config.latent_size).eval()
    utils.load_checkpoint(model, checkpoint_path, 'cpu')

    num_runs = 100
    mse_error = 0
    with torch.no_grad():
        for run_i in range(num_runs):
            states = pend_valid_data[run_i]
            # decoded, mu, logvar = model(states)
            mu, logvar = model.encode(states[0:2])
            decoded = model.decode(mu)

            z_next = model.f_trainsition(mu)
            decoded_next = model.decode(z_next)
            print('true state:    {}'.format(states[2:4]))
            print('decoded state: {}'.format(decoded_next))
            # mse_error += torch.pow((states - decoded), 2).sum()
        # mse_error = mse_error / num_runs
        # print('MSE Error: {}'.format(mse_error))


if __name__ == '__main__':
    # calculate_mse()
    # evaluate_temporal_vae()
    rollout_temporal_vae()
