import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from networks.action_conditional_lstm import ActionCondLSTM


def test_pendulum():

    # Load model and weights
    device = torch.device('cpu')
    model = ActionCondLSTM(input_size=3, action_size=1, hidden_size=16, num_layers=1).to(device)
    state_checkpoint_path = './checkpoints/checkpoint_5k_one_steps.pt'
    model.load_state_dict(torch.load(state_checkpoint_path, map_location=device), strict=True)
    model.eval()

    model_two_steps = ActionCondLSTM(input_size=3, action_size=1, hidden_size=16, num_layers=1).to(device)
    state_checkpoint_path = './checkpoints/checkpoint_5k_one_step_hidden_loss.pt'
    model_two_steps.load_state_dict(torch.load(state_checkpoint_path, map_location=device), strict=True)
    model_two_steps.eval()

    # Plotting
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    fig.suptitle('Pendulum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity')
    # ax.legend(('training', 'validation'))
    # plt.legend()

    # Environment
    env = gym.make('Pendulum-v0')
    state_true = env.reset()
    print('RESETTING!')
    state_pred = state_true
    state_pred_2 = state_true
    horizon_length = 150

    for i in range(horizon_length):
        a = env.action_space.sample()
        # env.render()
        state_action_t = np.array([[state_pred[0], state_pred[1], state_pred[2], a]], dtype=np.float32)
        state_action_t_2 = np.array([[state_pred_2[0], state_pred_2[1], state_pred_2[2], a]], dtype=np.float32)
        if i == 0:
            states_actions = state_action_t
            states_actions_2 = state_action_t_2
            states_true = np.array([state_true])
        else:
            states_actions = np.append(states_actions, state_action_t, axis=0)
            states_actions_2 = np.append(states_actions_2, state_action_t_2, axis=0)
            states_true = np.append(states_true, [next_state_true], axis=0)

        # nn model predictions
        states_actions_torch = torch.from_numpy(states_actions)
        states_actions_torch = states_actions_torch.view(1, -1, 4)
        states_actions_torch = states_actions_torch.to(device)

        states_actions_torch_2 = torch.from_numpy(states_actions_2)
        states_actions_torch_2 = states_actions_torch_2.view(1, -1, 4)
        states_actions_torch_2 = states_actions_torch_2.to(device)
        with torch.no_grad():
            next_state_pred, _, _ = model.forward(states_actions_torch[:, :, 0:3], states_actions_torch[:, :, 3:4])
            next_state_pred_2, _, _ = model_two_steps.forward(states_actions_torch_2[:, :, 0:3], states_actions_torch_2[:, :, 3:4])

        next_state_pred = next_state_pred.view(next_state_pred.size(1), next_state_pred.size(2))
        next_state_pred_numpy = next_state_pred[-1].cpu().numpy()
        state_pred = next_state_pred_numpy

        next_state_pred_2 = next_state_pred_2.view(next_state_pred_2.size(1), next_state_pred_2.size(2))
        next_state_pred_numpy_2 = next_state_pred_2[-1].cpu().numpy()
        state_pred_2 = next_state_pred_numpy_2

        # simulator
        next_state_true, r, d, _ = env.step(a)

        ax.plot(states_true[:, 2], c='r', label='true state', linewidth=2)
        ax.plot(states_actions[:, 2], '--', c='b', label='1-step w/o hidden loss', linewidth=2.5)
        ax.plot(states_actions_2[:, 2], '--', c='g', label='1-step w/ hidden loss', linewidth=2.5)
        plt.pause(0.05)

        if i == 0:
            plt.legend()

    plt.show()

def run_model():

    # Load the model
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_32h_2step.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=32, num_layers=1, bias=True)
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
    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': 4.0})
    action_spec = env.action_spec()
    env.physics.model.dof_damping[0] = 0.0
    time_step = env.reset()

    state_sim = np.hstack((time_step.observation['orientation'], time_step.observation['velocity']))
    state_net = state_sim.copy()
    state_net_2 = state_sim.copy()
    N = 200
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

    state_i = 0
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
    test_pendulum()
