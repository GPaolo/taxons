# Created by Giuseppe Paolo 
# Date: 17/07/2019

import numpy as np
from core.utils import utils
from baselines.baseline import BaseBaseline
import gc

class RandomBD(BaseBaseline):
  """
  Performs standard NS with handcrafted fetures
  """
  # ---------------------------------------------------
  def __init__(self, env, parameters):
    super().__init__(env, parameters)
    np.random.seed(self.params.seed)
  # ---------------------------------------------------

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
        agent_input = [obs, t/self.params.max_episode_len] # Observation and time. The time is used to see when to stop the action. TODO move the action stopping outside of the agent
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
    agent['features'] = [np.random.random(self.params.feature_size), None] #RBD uses random vectors as features to calculate the BD
    return cumulated_reward
  # ---------------------------------------------------

  # # ---------------------------------------------------
  # def train(self, steps=10000):
  #   for self.elapsed_gen in range(steps):
  #     for agent in self.population:
  #       self.evaluate_agent(agent)
  #
  #     max_rew = np.max(self.population['reward'].values)
  #     self.opt.step()
  #
  #     if self.elapsed_gen % 10 == 0:
  #       gc.collect()
  #       print('Seed {} - Generation {}'.format(self.params.seed, self.elapsed_gen))
  #       if self.archive is not None:
  #         print('Seed {} - Archive size {}'.format(self.params.seed, self.archive.size))
  #       print('Seed {} - Max reward {}'.format(self.params.seed, max_rew))
  #       print('Saving checkpoint...')
  #       self.save(ckpt=True)
  #       print("Done")
  #       print()
  #
  #     if self.archive is not None:
  #       bs_points = np.concatenate(self.archive['bs'].values)
  #     else:
  #       bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
  #     if 'Ant' in self.params.env_tag:
  #       u_limit = 3.5
  #       l_limit = -u_limit
  #     elif 'FastsimSimpleNavigation' in self.params.env_tag:
  #       u_limit = 600
  #       l_limit = 0
  #     else:
  #       u_limit = 1.35
  #       l_limit = -u_limit
  #
  #     coverage = utils.show(bs_points, filepath=self.save_path,
  #                           info={'gen':self.elapsed_gen, 'seed':self.params.seed},
  #                           upper_limit=u_limit, lower_limit=l_limit)
  #
  #     self.logs['Generation'].append(str(self.elapsed_gen))
  #     self.logs['Avg gen surprise'].append('0')
  #     self.logs['Max reward'].append(str(max_rew))
  #     self.logs['Archive size'].append(str(self.archive.size))
  #     self.logs['Coverage'].append(str(coverage))
  #     if self.END:
  #       print('Seed {} - Quitting.'.format(self.params.seed))
  #       break
  #   gc.collect()
  # # ---------------------------------------------------














