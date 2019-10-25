# Created by giuseppe
# Date: 27/09/19

import numpy as np
from core.utils import utils
from baselines.baseline import BaseBaseline
from skimage.transform import resize

class ImageBD(BaseBaseline):
  """
  Performs NS with high-dimensional BD (e.g. RGB images)
  """
  # ---------------------------------------------------
  def evaluate_agent(self, agent):
    """
    This function evaluates the agent in the environment.
    :param agent: agent to evaluate
    :return:
    """
    done = False
    cumulated_reward = 0

    obs = self.env.reset()
    t = 0
    while not done:
      if 'FastsimSimpleNavigation' in self.params.env_tag:
        agent_input = [t/self.params.max_episode_len, obs] # Observation and time. The time is used to see when to stop the action. TODO move the action stopping outside of the agent
        action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input))
      else:
        agent_input = t
        action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input/self.params.max_episode_len))

      obs, reward, done, info = self.env.step(action)
      t += 1
      cumulated_reward += reward

      if  t >= self.params.max_episode_len:
        done = True

      if 'Ant' in self.params.env_tag:
        CoM = np.array([self.env.robot.body_xyz[:2]])
        if np.any(np.abs(CoM) >= np.array([3, 3])):
          done = True

    try:
      state = self.env.render(mode='rgb_array', top_bottom=True)
    except:
      state = self.env.render(mode='rgb_array')
    state = state / np.max((np.max(state), 1))
    state = resize(state, (64, 64))

    agent['bs'] = utils.extact_hd_bs(self.env, obs, reward, done, info)
    agent['reward'] = cumulated_reward
    agent['features'] = [state.ravel(), None] #Here we use HD images as features to calculate the BD
    return cumulated_reward
  # ---------------------------------------------------
