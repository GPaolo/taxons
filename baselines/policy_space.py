# Created by Giuseppe Paolo 
# Date: 30/06/2019

import numpy as np
from core.utils import utils
from baselines.baseline import BaseBaseline
import gc

class PolicySpace(BaseBaseline):
  """
  Performs NS in the policy parameters space
  """
  # ---------------------------------------------------
  def evaluate_agent(self, agent):
    """
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    """
    done = False
    cumulated_reward = 0

    obs = self.env.reset()
    t = 0
    while not done:
      if 'FastsimSimpleNavigation' in self.params.env_tag:
        agent_input = [t/self.params.max_episode_len, obs] # Observation and time. The time is used to see when to stop the action.
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

    agent['bs'] = utils.extact_hd_bs(self.env, obs, reward, done, info)
    agent['reward'] = cumulated_reward

    # Extract genome as a feature
    feat = []
    for k in agent['agent'].genome:
      if isinstance(k, dict):
        for i in k:
          if i is not 'name':
            feat.append(np.array(k[i]).flatten())
      else:
        feat.append(np.array([k]))

    agent['features'] = [np.concatenate(np.array(feat)), None] #PS uses the genome as feature to calculate the BD
    return cumulated_reward
  # ---------------------------------------------------