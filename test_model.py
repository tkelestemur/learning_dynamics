import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from action_conditional_lstm import ActionCondLSTM


def test_pendulum():

    # Load model and weights
    device = torch.device('cpu')
    model = ActionCondLSTM(input_size=3, action_size=1, hidden_size=16, num_layers=1).to(device)
    state_checkpoint_path = './checkpoints/checkpoint_5k_one_steps.pt'
    model.load_state_dict(torch.load(state_checkpoint_path, map_location=device), strict=True)
    model.eval()

    model_two_steps = ActionCondLSTM(input_size=3, action_size=1, hidden_size=16, num_layers=1).to(device)
    state_checkpoint_path = './checkpoints/checkpoint_5k_two_steps.pt'
    model_two_steps.load_state_dict(torch.load(state_checkpoint_path, map_location=device), strict=True)
    model_two_steps.eval()

    # Plotting
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    fig.suptitle('Pendulum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Angle')
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
            next_state_pred = model.forward(states_actions_torch[:, :, 0:3], states_actions_torch[:, :, 3:4])
            next_state_pred_2 = model_two_steps.forward(states_actions_torch_2[:, :, 0:3], states_actions_torch_2[:, :, 3:4])

        next_state_pred = next_state_pred.view(next_state_pred.size(1), next_state_pred.size(2))
        next_state_pred_numpy = next_state_pred[-1].cpu().numpy()
        state_pred = next_state_pred_numpy

        next_state_pred_2 = next_state_pred_2.view(next_state_pred_2.size(1), next_state_pred_2.size(2))
        next_state_pred_numpy_2 = next_state_pred_2[-1].cpu().numpy()
        state_pred_2 = next_state_pred_numpy_2

        # simulator
        next_state_true, r, d, _ = env.step(a)

        ax.plot(states_true[:, 0], c='r', label='true state', linewidth=2)
        ax.plot(states_actions[:, 0], '--', c='b', label='1-step loss', linewidth=2.5)
        ax.plot(states_actions_2[:, 0], '--', c='g', label='2-step loss', linewidth=2.5)
        plt.pause(0.05)

        if i == 0:
            plt.legend()

    plt.show()


if __name__ == '__main__':
    test_pendulum()
