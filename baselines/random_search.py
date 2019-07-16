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
    super(BaseBaseline, self).__init__(env, parameters)

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

    obs = utils.obs_formatting(self.params.env_tag, self.env.reset())
    t = 0
    while not done:
      agent_input = t
      action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input))
      obs, reward, done, info = self.env.step(action)
      obs = utils.obs_formatting(self.params.env_tag, obs)
      t += 1
      cumulated_reward += reward

      if 'Ant' in self.params.env_tag:
        CoM = np.array([self.env.env.data.qpos[:2]])
        if t >= self.params.max_episode_len or np.any(np.abs(CoM) >= np.array([4, 4])):
          done = True

    if 'Ant' in self.params.env_tag:
      agent['bs'] = np.array([self.env.env.data.qpos[:2]])  # xy position of CoM of the robot
    else:
      agent['bs'] = np.array([[obs[0][0], obs[0][1]]])
    agent['reward'] = cumulated_reward

    agent['features'] = [None, None] #TODO check this!!!
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
      limit = 5
    else:
      limit = 1.35
    coverage = utils.show(bs_points, filepath=self.save_path, info={'gen': self.elapsed_gen, 'seed': self.params.seed},
                          limit=limit)

    self.logs['Generation'] = [str(self.elapsed_gen)] * self.params.generations
    self.logs['Avg gen surprise'] = ['0'] * self.params.generations
    self.logs['Max reward'] = [str(max_rew)] * self.params.generations
    self.logs['Archive size'] = [str(self.population.size)] * self.params.generations
    self.logs['Coverage'] = [str(coverage)] * self.params.generations
    gc.collect()
  # ---------------------------------------------------
