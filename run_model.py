from dm_control import suite
import torch
import numpy as np
import matplotlib.pyplot as plt
from networks.lstm_autoencoder import LSTMAutoEncoder
plt.style.use('ggplot')


def run_model():

    # Load the model
    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_5k.pth'
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
    model.eval()

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

    while not time_step.last():

        a = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

        if time_step.first():
            trajectory_sim = np.hstack((state_sim, a)).reshape(1, 4)
        else:
            traj_point_sim = np.hstack((state_sim, a)).reshape(1, 4)
            trajectory_sim = np.append(trajectory_sim, traj_point_sim, axis=0)

        trajectory_model_torch = torch.from_numpy(trajectory_sim)

        trajectory_model_torch = trajectory_model_torch.view(1, -1, 4).float()
        with torch.no_grad():
            _, decoded, _ = model.forward(trajectory_model_torch[:, :, 0:3], trajectory_model_torch[:, :, 3:4])

        trajectory_pred = decoded.view(decoded.size(1), decoded.size(2)).numpy()

        time_step = env.step(a)
        state_sim = np.hstack((time_step.observation['orientation'], time_step.observation['velocity']))

        ax.plot(trajectory_sim[:, 2], c='r', label='true state', linewidth=2)
        ax.plot(trajectory_pred[:, 2], '--', c='b', label='1-step w/o hidden loss', linewidth=2.5)
        plt.pause(0.05)

    plt.show()


if __name__ == '__main__':
    run_model()