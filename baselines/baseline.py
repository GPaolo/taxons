# Created by Giuseppe Paolo 
# Date: 16/07/2019

import numpy as np
from core.evolution import population, agents
from core.utils import utils
import os, gc, json

class BaseBaseline(object):
  """
  This is the base baseline from where to create all the other ones
  """
  # ---------------------------------------------------
  def __init__(self, env, parameters):
    """
    Constructor
    :param env: gym environment
    :param parameters: parameters of the run
    """
    self.params = parameters
    self.pop_size = self.params.pop_size
    self.env = env
    self.save_path = self.params.save_path
    self.agents_shapes = self.params.agent_shapes
    self.agent_name = self.params.qd_agent

    self.logs = utils.Logger({'Generation': [], 'Avg gen surprise': [], 'Max reward': [], 'Archive size': [], 'Coverage': []})

    if self.agent_name == 'Neural':
      self.agent_type = agents.FFNeuralAgent
    elif self.agent_name == 'DMP':
      self.agent_type = agents.DMPAgent

    self.population = population.Population(agent=self.agent_type,
                                            shapes=self.agents_shapes,
                                            pop_size=self.pop_size)
    self.archive = None
    if self.params.use_archive:
      self.archive = population.Population(agent=self.agent_type,
                                           shapes=self.agents_shapes,
                                           pop_size=0)

    self.opt = self.params.optimizer(self.population, archive=self.archive, mutation_rate=self.params.mutation_rate,
                                     metric_update_interval=self.params.update_interval)

    self.END = False
    self.elapsed_gen = 0
  # ---------------------------------------------------

  # ---------------------------------------------------
  def evaluate_agent(self, agent):
    """
    Evaluates the agent. This one needs to be reimplemented. It raises a NotImplementedError
    :param agent:
    :return:
    """
    raise NotImplementedError
  # ---------------------------------------------------

  # ---------------------------------------------------
  def train(self, steps=10000):
    """
    Main function to run.
    :param steps: number of steps to run the search.
    :return:
    """
    for self.elapsed_gen in range(steps):
      # Evaluate all the agents
      for agent in self.population:
        self.evaluate_agent(agent)

      max_rew = np.max(self.population['reward'].values)
      self.opt.step() # Perform optimization step, updating the archive and the population

      # Every 10 generation print an overview of the process, save a checkpoint and perform garbage collection
      if self.elapsed_gen % 10 == 0:
        gc.collect()
        print('Seed {} - Generation {}'.format(self.params.seed, self.elapsed_gen))
        if self.archive is not None:
          print('Seed {} - Archive size {}'.format(self.params.seed, self.archive.size))
        print('Seed {} - Max reward {}'.format(self.params.seed, max_rew))
        print('Saving checkpoint...')
        self.save(ckpt=True)
        print("Done")
        print()

      if self.archive is not None:
        bs_points = np.stack(self.archive['bs'].to_list())
      else:
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

      self.logs.register_log('Generation', self.elapsed_gen)
      self.logs.register_log('Avg gen surprise', 0)
      self.logs.register_log('Max reward', max_rew)
      self.logs.register_log('Archive size', self.archive.size)
      self.logs.register_log('Coverage', coverage)
      if self.END:
        print('Seed {} - Quitting.'.format(self.params.seed))
        break
    gc.collect()
    # ---------------------------------------------------

  # ---------------------------------------------------
  def save(self, ckpt=False):
    if ckpt:
      folder = 'models/ckpt'
    else:
      folder = 'models'
    save_subf = os.path.join(self.save_path, folder)
    print('Seed {} - Saving...'.format(self.params.seed))
    if not os.path.exists(save_subf):
      try:
        os.makedirs(os.path.abspath(save_subf))
      except:
        print('Seed {} - Cannot create save folder.'.format(self.params.seeds))
    self.population.save_pop(save_subf, 'pop')
    if self.archive is not None:
      self.archive.save_pop(save_subf, 'archive')

    self.logs.save(self.save_path)
    print('Seed {} - Done'.format(self.params.seed))
  # ---------------------------------------------------