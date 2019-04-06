import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def plot_loss():
    loss = np.genfromtxt('./loss/loss_5k.csv', delimiter=',')
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Pendulum [Training: 5k - Validation: 1k]')
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Loss ||$f(s_t, a_t) - s_{t+1}||$')
    ax.legend(('training', 'validation'))

    ax.plot(loss[:, 0], linewidth=3)
    ax.plot(loss[:, 1], '--', linewidth=3)
    # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_loss()