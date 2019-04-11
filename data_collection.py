from dm_control import suite
from dm_control import viewer
import numpy as np



# Load one task:
env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': 5.0})
env.physics.model.dof_damping[0] = 0.0
# env = ActionRepeat(env, 8)
env.reset()




# env = ActionRepeat(env, 8)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()


def random_policy(time_step):
  # del time_step
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  print(env.physics.data.ctrl[:])
  # print(action)
  # action = 1
  return action


viewer.launch(env, policy=random_policy)
