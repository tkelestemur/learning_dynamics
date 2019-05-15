from dm_control import suite
from dm_control import viewer
from tqdm import tqdm
import numpy as np


def view_env():
    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': 4.0})
    env.physics.model.dof_damping[0] = 0.0
    # env.physics.model.timestep = 1
    # env.n_sub_steps = 100
    # env.physics.model.actuator_ctrllimited[0] = False
    # time_step = env.reset()

    action_spec = env.action_spec()

    def random_policy(time_step):
        # del time_step
        # a = np.random.uniform(-5.0, 5.0, size=action_spec.shape)
        # if env.physics.data.qpos[0] < 0.0:
        #     print('qpos {} qvel {}'.format(2*np.pi + env.physics.data.qpos[0], env.physics.data.qvel[0]))
        # else:
        #     print('qpos {} qvel {}'.format(env.physics.data.qpos[0], env.physics.data.qvel[0]))

        a = 0.2
        print(env.physics.data.time)
        return a

    def no_action_poilciy(time_step):
        # print(np.hstack((env.physics.named.data.qpos['hinge'], env.physics.named.data.qvel['hinge'])))
        print(env.physics.named.data.qvel['hinge'])
        # print('cos sin : {}'.format(time_step.observation['orientation']))
        return 0.0

    viewer.launch(env, policy=no_action_poilciy)


def collect_pendulum_data():

    num_runs = 1
    traj_horizon = 2.0

    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': traj_horizon})
    env.physics.model.dof_damping[0] = 0.0
    # env.physics.model.timestep = 0.05
    action_spec = env.action_spec()

    trajectories = []
    for run_i in tqdm(range(num_runs)):
        time_step = env.reset()
        trajectory = []
        while not time_step.last():
            # action = np.random.uniform(-5.0, 5.0, size=action_spec.shape)
            action = 0
            # action = np.random.uniform(action_spec.minimum,
                                       # action_spec.maximum,
                                       # size=action_spec.shape)

            # state_action_t = np.hstack((env.physics.data.qpos[0], time_step.observation['orientation'],
                                        # time_step.observation['velocity'], action))
            state_action_t = time_step.observation['orientation']
            trajectory += [state_action_t]
            time_step = env.step(action)

        trajectory = np.array(trajectory)
        trajectories += [trajectory]

    trajectories = np.array(trajectories)

    np.save('./pend_data/pendulum_no_action_single_run', trajectories)

def pendulum_no_acition_data():
    num_runs = 110000
    env = suite.load(domain_name="pendulum", task_name="swingup")

    states = []
    for run_i in tqdm(range(num_runs)):

        # time_step = env.reset()
        # next_time_step = env.step(0.0)

        # state_init = np.hstack((env.physics.named.data.qpos['hinge'], env.physics.named.data.qvel['hinge']))
        # state_next = np.hstack((env.physics.named.data.qpos['hinge'], env.physics.named.data.qvel['hinge']))

        time_step = env.reset()
        next_time_step = env.step(0.0)

        # state_init = np.hstack((time_step.observation['orientation'], time_step.observation['velocity']))
        # state_next = np.hstack((next_time_step.observation['orientation'], next_time_step.observation['velocity']))

        # state = np.hstack((state_init, state_next))
        state = np.hstack((time_step.observation['orientation'], next_time_step.observation['orientation']))
        states += [state]

    states = np.array(states)
    np.save('./pend_data/pendulum_no_action_bounded', states)

if __name__ == '__main__':
    collect_pendulum_data()
    # pendulum_no_acition_data()
    # view_env()
