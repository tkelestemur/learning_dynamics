from dm_control import suite
from dm_control import viewer
from tqdm import tqdm
import numpy as np
import torch


def view_env():

    env = suite.load(domain_name="pendulum", task_name="swingup")
    action_spec = env.action_spec()

    def no_action_policy(time_step):
        del time_step
        return np.zeros(2)

    def random_action_policy(time_step):
        del time_step
        return np.random.uniform(low=action_spec.minimum,
                                 high=action_spec.maximum,
                                 size=action_spec.shape)

    viewer.launch(env, policy=random_action_policy)


def generate_trajectories():

    num_runs = 1
    horizon = 2.0

    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': horizon})

    num_steps = int(horizon / env.physics.model.opt.timestep)
    trajectories = np.empty((num_runs, num_steps, 3), dtype=np.float32)
    for run_i in tqdm(range(num_runs)):
        time_step = env.reset()
        trajectory = np.empty((num_steps, 3), dtype=np.float32)
        for step_i in range(num_steps):
            action = 0.0
            trajectory[step_i] = np.hstack((time_step.observation['orientation'], time_step.observation['velocity']))
            # for t in range(4):
            time_step = env.step(action)

        trajectories[run_i] = trajectory

    trajectories = np.array(trajectories)

    np.save('./pend_data/pendulum_no_action_bounded_trajectory', trajectories)


def generate_states():
    num_runs = 10000
    env = suite.load(domain_name="pendulum", task_name="swingup")

    data = np.empty((num_runs, 6))
    for run_i in tqdm(range(num_runs)):

        time_step = env.reset()
        state_init = time_step.observation
        # for i in range(4):
        time_step = env.step(0.0)
        state_next = time_step.observation

        data[run_i] = np.hstack((state_init['orientation'], state_init['velocity'],
                                 state_next['orientation'], state_next['velocity']))

    np.save('./pend_data/pendulum_no_action_bounded_valid', data)


if __name__ == "__main__":
    # generate_trajectories()
    # generate_states()
    view_env()
