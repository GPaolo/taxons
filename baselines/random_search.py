# Created by Giuseppe Paolo 
# Date: 08/07/2019

import numpy as np
from core.evolution import population
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
    self.archive = population.Population(agent=self.agent_type,
                                           shapes=self.agents_shapes,
                                           pop_size=0)
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

    agent['features'] = [None, None] # RS does not uses any features cuz it does not do any evolution
    return cumulated_reward
  # ---------------------------------------------------

  # ---------------------------------------------------
  def train(self, *args, **kwargs):
    for idx, agent in enumerate(self.population):
      self.evaluate_agent(agent)
      self.archive.add(self.population.copy(idx, with_data=True))
      if idx % 100 == 0:
        gc.collect()
        print('Seed {} - Agent {}'.format(self.params.seed, idx))

      # Every 5 agents there is a generation. This is done to keep the logs consistent with the other experiments
      # We do idx + 1 cause idx goes from 0 not from 1
      if (idx + 1) % 5 == 0 and idx != 0:
        self.elapsed_gen += 1

        bs_points = np.concatenate(self.archive['bs'].values)
        if 'Ant' in self.params.env_tag:
          u_limit = 3.5
          l_limit = -u_limit
        elif 'FastsimSimpleNavigation' in self.params.env_tag:
          u_limit = 600
          l_limit = 0
        else:
          u_limit = 1.35
          l_limit = -u_limit

        max_rew = np.max(self.archive['reward'].values)
        coverage = utils.show(bs_points, filepath=self.save_path,
                              info={'gen': self.elapsed_gen, 'seed': self.params.seed},
                              upper_limit=u_limit, lower_limit=l_limit)

        self.logs['Generation'].append(str(self.elapsed_gen))
        self.logs['Avg gen surprise'].append('0')
        self.logs['Max reward'].append(str(max_rew))
        self.logs['Archive size'].append(str(self.archive.size))
        self.logs['Coverage'].append(str(coverage))
      if self.END:
        print('Seed {} - Quitting.'.format(self.params.seed))
        break

    gc.collect()
  # ---------------------------------------------------
