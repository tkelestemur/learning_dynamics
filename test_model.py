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
    model.load_state_dict(torch.load('./checkpoints/checkpoint_5k.pt'), strict=True)
    model.eval()

    # Plotting
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Pendulum')
    ax.set_xlabel('timestep')
    ax.set_ylabel('velocity')
    ax.legend(['simulation'])

    # Environment
    env = gym.make('Pendulum-v0')
    state_true = env.reset()
    print('RESETTING!')
    state_pred = state_true
    horizon_length = 100

    for i in range(horizon_length):
        a = env.action_space.sample()
        # env.render()
        state_action_t = np.array([[state_pred[0], state_pred[1], state_pred[2], a, 0.0, 0.0, 0.0]], dtype=np.float32)
        if i == 0:
            states_actions = state_action_t
            states_true = np.array([state_true])
        else:
            states_actions = np.append(states_actions, state_action_t, axis=0)
            states_true = np.append(states_true, [next_state_true], axis=0)
        # nn model prediction

        states_actions_torch = torch.from_numpy(states_actions)
        states_actions_torch = states_actions_torch.view(1, -1, 7)
        states_actions_torch = states_actions_torch.to(device)
        with torch.no_grad():
            next_state_pred = model.forward(states_actions_torch)

        next_state_pred = next_state_pred.view(next_state_pred.size(1), next_state_pred.size(2))
        next_state_pred_numpy = next_state_pred[-1].cpu().numpy()

        state_pred = next_state_pred_numpy
        # states_pred[i] = next_state_pred_numpy
        #
        # # simulator
        next_state_true, r, d, _ = env.step(a)

        ax.plot(states_true[:, 0], c='r')
        ax.plot(states_actions[:, 0], c='b')
        plt.pause(0.05)

    plt.show()


if __name__ == '__main__':
    test_pendulum()
