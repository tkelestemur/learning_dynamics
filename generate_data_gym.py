import gym
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


if __name__ == '__main__':
    view_env()