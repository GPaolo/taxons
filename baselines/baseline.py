# Created by Giuseppe Paolo 
# Date: 16/07/2019

import numpy as np
from core.qd import population, agents
from core.utils import utils
import os, gc, json

class BaseBaseline(object):
  """
  This is the base baseline from where to create all the other ones
  """
  # ---------------------------------------------------
  def __init__(self, env, parameters):
    self.params = parameters
    self.pop_size = self.params.pop_size
    self.env = env
    self.save_path = self.params.save_path
    self.agents_shapes = self.params.agent_shapes
    self.agent_name = self.params.qd_agent

    self.logs = {'Generation': [], 'Avg gen surprise': [], 'Max reward': [], 'Archive size': [], 'Coverage': []}

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
    raise NotImplementedError
  # ---------------------------------------------------

  # ---------------------------------------------------
  def train(self, *args, **kwargs):
    raise NotImplementedError
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

    with open(os.path.join(self.save_path, 'logs.json'), 'w') as f:
      json.dump(self.logs, f, indent=4)
    print('Seed {} - Done'.format(self.params.seed))
  # ---------------------------------------------------