# Created by Giuseppe Paolo 
# Date: 08/07/2019

import numpy as np
from core.qd import population
from core.utils import utils
from baselines.baseline import BaseBaseline
import gc


class RandomSearch(BaseBaseline):
  """Generates a lot of random agents and tests them"""
  # ---------------------------------------------------
  def __init__(self, env, parameters):
    super().__init__(env, parameters)

    self.pop_size = self.params.generations*5
    self.params.pop_size = self.pop_size
    self.population = population.Population(agent=self.agent_type,
                                            shapes=self.agents_shapes,
                                            pop_size=self.pop_size)
    self.archive = None
    self.opt = self.params.optimizer(self.population, archive=self.archive, mutation_rate=self.params.mutation_rate,
                                     metric_update_interval=self.params.update_interval)
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

    agent['features'] = [None, None] #TODO check this!!! # RS does not uses any features cuz it does not do any evolution
    return cumulated_reward
  # ---------------------------------------------------

  # ---------------------------------------------------
  def train(self, *args, **kwargs):
    for idx, agent in enumerate(self.population):
      self.evaluate_agent(agent)
      if idx % 100 == 0:
        gc.collect()
        print('Seed {} - Agent {}'.format(self.params.seed, idx))

    max_rew = np.max(self.population['reward'].values)

    bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
    if 'Ant' in self.params.env_tag:
      u_limit = 3.5
      l_limit = -u_limit
    elif 'FastsimSimpleNavigation' in self.params.env_tag:
      u_limit = 600
      l_limit = 0
    else:
      u_limit = 1.35
      l_limit = -u_limit

    coverage = utils.show(bs_points, filepath=self.save_path,
                          info={'gen': self.elapsed_gen, 'seed': self.params.seed},
                          upper_limit=u_limit, lower_limit=l_limit)

    self.logs['Generation'] = [str(self.elapsed_gen)] * self.params.generations
    self.logs['Avg gen surprise'] = ['0'] * self.params.generations
    self.logs['Max reward'] = [str(max_rew)] * self.params.generations
    self.logs['Archive size'] = [str(self.population.size)] * self.params.generations
    self.logs['Coverage'] = [str(coverage)] * self.params.generations
    gc.collect()
  # ---------------------------------------------------
