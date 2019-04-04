import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def plot_loss():
    loss = np.genfromtxt('./loss/loss_5k.csv', delimiter=',')
    print(loss.shape)
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('[Training: 5k - Validation: 1k]')
    # fig.set_size_inches(12, 8)
    ax.plot(loss[:, 0], linewidth=3)
    ax.plot(loss[:, 1], '--', linewidth=3)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Loss')
    ax.legend(('training', 'validation'))

    # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_loss()