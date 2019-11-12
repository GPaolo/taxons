# Created by giuseppe
# Date: 28/03/19

import os
from core.utils import utils, optimizer
import json

class Params(object):
  # ---------------------------------------------------------
  def __init__(self):
    # Main parameters
    # ----------------------
    self.info = 'Maze with 1k gens. NT'
    self.exp_name = 'Maze_NT'

    self.exp = 'TAXONS' # 'TAXONS', 'TAXON', 'TAXOS', 'NT, 'NS', 'PS', 'RS', 'RBD', 'IBD'
    self.env_tag = 'FastsimSimpleNavigation-v0' # Billiard-v0 AntMuJoCoEnv-v0 FastsimSimpleNavigation-v0
    self.threads = 4
    # ----------------------

    self.set_env_params()
    self.set_exp_params()

    # Other params
    self.save_path = os.path.join(utils.get_projectpath(), 'experiments', self.exp_name)
    self.seed = 7
    self.parallel = True

    self.pop_size = 100
    self.use_archive = True
    self.mutation_rate = 0.9

    # Metric
    self.metric = 'AE'  # 'RND', 'BVAE', 'FFAE', 'AE'
    self.feature_size = 10
    self.learning_rate = 0.001  # 0.0001 for RND
    self.lr_scale_fact = 0.5
    self.per_agent_update = False
    self.train_on_archive = True
    self.update_interval = 30

  # ---------------------------------------------------------

  # ---------------------------------------------------------
  def set_env_params(self):
    if 'Ant' in self.env_tag:
      self.max_episode_len = 300
      self.agent_shapes = {'dof': 8, 'degree': 5, 'type': 'sin'}
      self.generations = 500
    elif 'FastsimSimpleNavigation' in self.env_tag:
      self.max_episode_len = 2000
      self.agent_shapes = {'input_shape': 5, 'output_shape': 2}
      self.generations = 1000
    elif 'Billiard' in self.env_tag:
      self.max_episode_len = 300
      self.agent_shapes = {'dof': self.action_shape, 'degree': 5, 'type': 'poly'}
      self.generations = 2000
    else:
      raise ValueError('Wrong environment name chosen')
  # ---------------------------------------------------------

  # ---------------------------------------------------------
  def set_exp_params(self):
    if self.exp == 'TAXONS':
      self.optimizer = optimizer.NoveltySurpriseOptimizer
      self.update_metric = True
      self.gpu = True
    elif self.exp == 'TAXON':
      self.optimizer = optimizer.NoveltyOptimizer
      self.update_metric = True
      self.gpu = True
    elif self.exp == 'TAXOS':
      self.optimizer = optimizer.SurpriseOptimizer
      self.update_metric = True
      self.gpu = True
    elif self.exp == 'NT':
      self.optimizer = optimizer.NoveltyOptimizer
      self.update_metric = False
      self.gpu = True
    else: # These are the baselines
      self.optimizer = optimizer.NoveltyOptimizer
      self.update_metric = True
      self.gpu = True
  # ---------------------------------------------------------


  # -----------------------------------------Setup----------------

  def _get_dict(self):
    params_dict = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
    del params_dict['optimizer']
    return params_dict
  # ---------------------------------------------------------

  def save(self):
    # if not os.path.exists(self.save_path):
    os.makedirs(self.save_path, exist_ok=True)
    with open(os.path.join(self.save_path, 'params.json'), 'w') as f:
      json.dump(self._get_dict(), f, indent=4)
  # ---------------------------------------------------------

  def load(self, load_path):
    assert os.path.exists(load_path), 'Specified parameter file does not exists in {}.'.format(load_path)
    with open(load_path) as f:
      data = json.load(f)
    # data['_seed'] = 1190
    for key in data:
      setattr(self, key, data[key])
      assert self.__dict__[key] == data[key], 'Could not set {} parameter.'.format(key)
    self._load_optimizer()
    print('Done')
  # ---------------------------------------------------------

  # Seed Property
  # ---------------------------------------------------------
  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, seed):
    self._seed = seed
    self.save_path = os.path.join(utils.get_projectpath(), 'experiments', self.exp_name, str(self.seed))
  # ---------------------------------------------------------
# ---------------------------------------------------------
