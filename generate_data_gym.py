import gym
import numpy as np
from tqdm import tqdm
import torch


env = gym.make('InvertedPendulum-v2')


def view_env():

    for i in range(10):
        done = False
        env.reset()
        while not done:
            obs, reward, done, _ = env.step(0.0)
            print(obs)
            env.render()


def generate_rollouts():

    num_runs = 2
    rollouts = []
    for run_i in tqdm(range(num_runs)):
        rollout = []
        obs, done = env.reset(), False
        t = 0
        while not done:
            rollout.append(obs.tolist())
            obs, reward, done, _ = env.step(0.0)

        rollouts.append(rollout)

    rollouts = np.array(list(rollouts))
    print(rollouts.shape)


if __name__ == '__main__':
    generate_rollouts()
    # view_env()