import torch
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')


def plot_mse():

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    ax.set_title('Pendulum Test Set [1K Run]')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Prediction MSE')

    results = [
                # './results/mse_16h_3step_5000_epochs_best.pt',
               # './results/mse_32h_1step_5000_epochs_best.pt',
               # './results/mse_32h_3step_5000_epochs_best.pt',
               # './results/mse_64h_3step_5000_epochs_best.pt',
               './results/mse_128h_3step_5000_epochs_best.pt',
               # './results/mse_256h_3step_5000_epochs_best.pt',
               # './results/mse_512h_3step_5000_epochs_best.pt'
               './results/mse_128h_3step_5000_epochs_linear_best.pt'
               ]

    labels = ['128h - w/ nonlinaer autoencoder', '128h - w/o nonlinear autoencoder']
    for i, result in enumerate(results):
        mse = torch.load(result)
        mean = torch.mean(mse, dim=0)
        std = torch.std(mse, dim=0)

        ax.plot(mean.numpy(), label=labels[i], linewidth=2)
        ax.fill_between(range(mse.size(1)), (mean-std).numpy(), (mean+std).numpy(), alpha=0.2)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_mse()
