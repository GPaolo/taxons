# Created by giuseppe
# Date: 28/03/19

import os
from core.utils import utils, optimizer
import json

class Params(object):
  def __init__(self):
    self.info = 'Ant with randomly assigned array as behaviour descriptor'

    self.exp_name = 'Collectdata_Ant_RBD'
    # Save Path
    self.save_path = os.path.join(utils.get_projectpath(), 'experiments', self.exp_name)
    self.seed = 7
    self.parallel = False
    self.baseline = 'RBD' # None, 'NS', 'PS', 'RS', 'RBD'

    # Environment
    # ---------------------------------------------------------
    self.env_tag = 'FastsimSimpleNavigation-v0' # Billiard-v0 Ant-v2 FastsimSimpleNavigation-v0
    self.max_episode_len = 300
    # ---------------------------------------------------------

    # QD
    # ---------------------------------------------------------
    self.generations = 500
    self.pop_size = 100
    self.use_archive = True
    self.mutation_rate = 0.9

    self.qd_agent = 'DMP'  # 'DMP
    if self.qd_agent == 'Neural':
      self.agent_shapes = {'input_shape': 1, 'output_shape': self.action_shape}
    elif self.qd_agent == 'DMP':
      self.agent_shapes = {'dof': self.action_shape, 'degree': 5, 'type': 'sin'} # poly, exp, sin
    # ---------------------------------------------------------

    # Metric
    # ---------------------------------------------------------
    self.gpu = True
    self.metric = 'AE'  # 'RND', 'BVAE', 'FFAE', 'AE'
    self.feature_size = 10
    self.learning_rate = 0.001 # 0.0001 for RND
    self.lr_scale_fact = 0.5
    self.per_agent_update = False
    self.update_metric = True
    self.train_on_archive = True
    self.update_interval = 30
    # ---------------------------------------------------------

    # Optimizer
    # ---------------------------------------------------------
    self.optimizer_type = 'Novelty' # 'Surprise', 'Pareto', 'Novelty'
    self._load_optimizer()
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
    data['_seed'] = 1190
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

  # Load optimizer
  def _load_optimizer(self):
    if self.optimizer_type == 'Novelty':
      self.optimizer = optimizer.NoveltyOptimizer
    elif self.optimizer_type == 'Surprise':
      self.optimizer = optimizer.SurpriseOptimizer
    elif self.optimizer_type == 'Pareto':
      self.optimizer = optimizer.ParetoOptimizer
  # ---------------------------------------------------------

  # Select action size
  @property
  def action_shape(self):
    if 'Ant' in self.env_tag:
      return 8
    elif 'Billiard' in self.env_tag:
      return 2
    elif 'MountainCar' in self.env_tag:
      return 2
  # ---------------------------------------------------------
# ---------------------------------------------------------
