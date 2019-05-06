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
               './results/mse_128h_3step_5000_epochs_best.pt',
               # './results/mse_16h_3step_5000_epochs_last.pt',
               # './results/mse_32h_1step_5000_epochs_last.pt',
               # './results/mse_32h_3step_5000_epochs_last.pt',
               './results/mse_128h_3step_5000_epochs_last.pt'
               ]
    for result in results:
        mse = torch.load(result)
        mean = torch.mean(mse, dim=0)
        std = torch.std(mse, dim=0)

        ax.plot(mean.numpy(), label=result.split('/')[-1], linewidth=2)
        # ax.fill_between(range(mse.size(1)), (mean-std).numpy(), (mean+std).numpy(), alpha=0.2)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_mse()
