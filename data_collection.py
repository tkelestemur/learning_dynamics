from dm_control import suite
from dm_control import viewer
import numpy as np


class ActionRepeat(object):
  """Repeat the agent action multiple steps."""

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      observ, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return observ, total_reward, done, info


# Load one task:
env = suite.load(domain_name="pendulum", task_name="swingup", environment_kwargs={'n_sub_steps': 10})
env.reset()
# e


# env = ActionRepeat(env, 8)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()


def random_policy(time_step):
  # del time_step
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  # print(action)
  # action = 20
  return action


viewer.launch(env, policy=random_policy)
