import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def plot_loss():
    loss_one_step = np.genfromtxt('./loss/lstm_auto_encoder/loss_16h_1step_linear.csv', delimiter=',')
    loss_two_step = np.genfromtxt('./loss/lstm_auto_encoder/loss_16h_1step_linear.csv', delimiter=',')
    fig, axes = plt.subplots(2, 1)
    # fig.suptitle('[2 Step Prediction - Training: 5k - Validation: 1k]')
    fig.set_size_inches(12, 8)
    axes[0].set_title('1 Step Prediction')
    # axes[0].set_xlabel('Number of Epochs')
    # axes[0].set_ylabel('Loss ||$f(s_t, a_t) - s_{t+1}||$')
    axes[0].set_ylabel('Loss')
    axes[0].legend(('training', 'validation'))

    axes[0].plot(loss_one_step[:, 0], linewidth=3)
    axes[0].plot(loss_one_step[:, 1], '--', linewidth=3)

    axes[1].set_title('2 Step Prediction')
    axes[1].set_xlabel('Number of Epochs')
    # axes[0].set_ylabel('Loss ||$f(s_t, a_t) - s_{t+1}||$')
    axes[1].set_ylabel('Loss')
    axes[1].legend(('training', 'validation'))

    axes[1].plot(loss_two_step[:, 0], linewidth=3)
    axes[1].plot(loss_two_step[:, 1], '--', linewidth=3)
    # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_loss()