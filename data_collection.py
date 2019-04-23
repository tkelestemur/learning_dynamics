from dm_control import suite
from dm_control import viewer
from tqdm import tqdm
import numpy as np


def view_env():
    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': 4.0})
    env.physics.model.dof_damping[0] = 0.0
    time_step = env.reset()

    action_spec = env.action_spec()

    def random_policy(time_step):
        # del time_step
        a = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        if env.physics.data.qpos[0] < 0.0:
            print('qpos {} qvel {}'.format(2*np.pi + env.physics.data.qpos[0], env.physics.data.qvel[0]))
        else:
            print('qpos {} qvel {}'.format(env.physics.data.qpos[0], env.physics.data.qvel[0]))
        # print(time_step.observation['orientation'])
        print(a[0])
        return a

    viewer.launch(env, policy=random_policy)


def collect_pendulum_data():

    num_runs = 1000
    traj_horizon = 4.0

    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': traj_horizon})
    env.physics.model.dof_damping[0] = 0.0
    action_spec = env.action_spec()

    trajectories = []
    for run_i in tqdm(range(num_runs)):
        time_step = env.reset()
        trajectory = []
        while not time_step.last():
            action = np.random.uniform(action_spec.minimum,
                                       action_spec.maximum,
                                       size=action_spec.shape)
            if env.physics.data.qpos[0] < 0.0:

                state_action_t = np.array([2*np.pi + env.physics.data.qpos[0], env.physics.data.qvel[0], action[0]])
            else:
                state_action_t = np.array([env.physics.data.qpos[0], env.physics.data.qvel[0], action])
            # print(state_action_t)
            # state_action_t = np.hstack((time_step.observation['orientation'], time_step.observation['velocity'], action))
            trajectory += [state_action_t]

            time_step = env.step(action)

        trajectory = np.array(trajectory)
        trajectories += [trajectory]

    trajectories = np.array(trajectories)

    np.save('./pend_data/pendulum_200_step_1k_run_valid_new2', trajectories)


if __name__ == '__main__':
    collect_pendulum_data()
    # view_env()
