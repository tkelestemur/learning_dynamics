import gym
import numpy as np


def pendulum():
    env = gym.make('Pendulum-v0')

    num_runs = 1000
    num_steps = 100
    actions = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    trajectories = []
    for i in range(num_runs):
        obs = env.reset()
        traj = []
        for j in range(num_steps):
            env.render()

            a = env.action_space.sample()
            # a_idx = np.random.randint(0, 5)
            # a = actions[a_idx]
            # a_onehot = np.zeros(5)
            # a_onehot[a_idx] = 1

            # data_t = np.array([obs[0], obs[1], obs[2], a])
            # data_t = np.hstack((data_t, a_onehot))
            # for k in range(1):
            #     obs, reward, done, _ = env.step(np.array([a]))
            next_obs, reward, done, _ = env.step(a)
            data_t = np.hstack((obs, a, next_obs))
            traj += [data_t]
            obs = next_obs

        traj = np.array(traj)
        print('Run : {}'.format(i))

        trajectories += [traj]
    trajectories = np.array(trajectories)
    print(trajectories.shape)

    # np.save('./pend_data/pendulum_100H_1000N.npy', trajectories)
    env.close()


def inverted_pendulum():
    env = gym.make('InvertedPendulum-v2')

    num_runs = 6000
    num_steps = 100

    trajectories = []
    for i in range(num_runs):
        obs = env.reset()
        traj = []
        T = 0
        for j in range(num_steps+1):
            env.render()

            a = env.action_space.sample()

            # data_t = np.array([obs[0], obs[1], obs[2]])
            # data_t = np.hstack((data_t, a_onehot))
            obs, reward, done, _ = env.step(a)
            T += 1
            if done:
                break
            # traj += [data_t]
        print('Trajecotry length: {}'.format(T))
        # traj = np.array(traj)
        print('Run : {}'.format(i))

        # trajectories += [traj]
    # trajectories = np.array(trajectories)
    # print(trajectories.shape)

    # np.save('./pend_data/pendulum_6k_disc.npy', trajectories)
    env.close()


if __name__ == '__main__':
    pendulum()
    # inverted_pendulum()