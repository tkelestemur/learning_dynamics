import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def plot_loss():
    loss_one_step = np.genfromtxt('./loss/lstm_auto_encoder/loss_128h_3step_nonlinear_5k_epochs.csv', delimiter=',')
    loss_two_step = np.genfromtxt('./loss/lstm_auto_encoder/loss_128h_3step_nonlinear_5k_epochs.csv', delimiter=',')
    fig, axes = plt.subplots(2, 1)
    # fig.suptitle('[2 Step Prediction - Training: 5k - Validation: 1k]')
    fig.set_size_inches(12, 8)
    axes[0].set_title('128h')
    # axes[0].set_xlabel('Number of Epochs')
    # axes[0].set_ylabel('Loss ||$f(s_t, a_t) - s_{t+1}||$')
    axes[0].set_ylabel('Loss')
    axes[0].legend(('training', 'validation'))

    axes[0].plot(loss_one_step[100:, 0], linewidth=3)
    axes[0].plot(loss_one_step[100:, 1], '--', linewidth=3)

    axes[1].set_title('256h')
    axes[1].set_xlabel('Number of Epochs')
    # axes[0].set_ylabel('Loss ||$f(s_t, a_t) - s_{t+1}||$')
    axes[1].set_ylabel('Loss')
    axes[1].legend(('training', 'validation'))

    axes[1].plot(loss_two_step[100:, 0], linewidth=3)
    axes[1].plot(loss_two_step[100:, 1], '--', linewidth=3)
    # fig.tight_layout()
    plt.show()

def plot_temporal_vae():
    loss = np.genfromtxt('./loss/temporal_vae/16h_4l_0.01beta.csv', delimiter=',')

    fig, axes = plt.subplots(2, 1)
    # fig.suptitle('beta = 0.0001')

    fig.set_size_inches(12, 8)
    beta_str = str(0.01)
    axes[0].set_title('Training Set - beta: ' + beta_str)
    axes[0].set_ylabel('Loss')

    axes[0].plot(loss[:, 0], label='total', linewidth=2)
    axes[0].plot(loss[:, 1], label='reconstruction', linewidth=2)
    axes[0].plot(loss[:, 2], label='prediction', linewidth=2)
    axes[0].plot(loss[:, 3], label='kl divergence', linewidth=2)

    axes[1].set_title('Validation Set - beta: ' + beta_str)
    axes[1].set_xlabel('Number of Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend(('training', 'validation'))

    axes[1].plot(loss[:, 4], label='total', linewidth=2)
    axes[1].plot(loss[:, 5], label='reconstruction', linewidth=2)
    axes[1].plot(loss[:, 6], label='prediction', linewidth=2)
    axes[1].plot(loss[:, 7], label='kl divergence', linewidth=2)

    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    # plot_loss()
    plot_temporal_vae()
