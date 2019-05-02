import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def plot_mse():

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    ax.set_title('Pendulum Test Set [1K Run]')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Prediction MSE')

    mse_16h_1step = np.load('./results/checkpoint_16h_3step_single_lstm.npy')
    mse_16h_1step_mean = np.mean(mse_16h_1step, axis=0)
    mse_16h_1step_std = np.std(mse_16h_1step, axis=0)

    # mse_16h_2step = np.load('./results/checkpoint_16h_2step_mse.npy')
    # mse_16h_2step_mean = np.mean(mse_16h_2step, axis=0)
    # mse_16h_2step_std = np.std(mse_16h_2step, axis=0)

    mse_16h_3step = np.load('./results/checkpoint_16h_3step_mse.npy')
    mse_16h_3step_mean = np.mean(mse_16h_3step, axis=0)
    mse_16h_3step_std = np.std(mse_16h_3step, axis=0)

    ax.plot(mse_16h_1step_mean, c='r', label='16 hidden - 3 step single lstm', linewidth=2)
    ax.fill_between(range(mse_16h_1step_mean.shape[0]), mse_16h_1step_mean-mse_16h_1step_std, mse_16h_1step_mean+mse_16h_1step_std, alpha=0.2, color='r')

    # ax.plot(mse_16h_2step_mean, c='b', label='16 hidden - 2 step', linewidth=2)
    # ax.fill_between(range(mse_16h_2step_mean.shape[0]), mse_16h_2step_mean-mse_16h_2step_std, mse_16h_2step_mean+mse_16h_2step_std, alpha=0.2, color='b')

    ax.plot(mse_16h_3step_mean, c='g', label='16 hidden - 3 step multiple lstm', linewidth=2)
    ax.fill_between(range(mse_16h_3step_mean.shape[0]), mse_16h_3step_mean-mse_16h_3step_std, mse_16h_3step_mean+mse_16h_3step_std, alpha=0.2, color='g')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_mse()
