# Created by Giuseppe Paolo
# Date: 27/02/19

import os
from core.utils import utils
import json

class Params(object):

  def __init__(self):
    '''
    This class is used as the default parameters collector to pass to the alg.
    '''

    self.info = 'Metric with AE. The metric is updated once per gen. AE has a single layer encoder and a single layer decoder'

    self.exp_name = 'test_params'
    self.seed = 7

    # Environment
    # ---------------------------------------------------------
    self.action_shape = 2
    self.env_tag = 'Billiard-v0'  # MountainCarContinuous-v0'
    # ---------------------------------------------------------

    # QD
    # ---------------------------------------------------------
    self.generations = 500
    self.pop_size = 100
    self.use_novelty = True
    self.use_archive = True

    self.qd_agent = 'Neural'  # 'DMP
    if self.qd_agent == 'Neural':
      self.agent_shapes = {'input_shape': 6, 'output_shape': self.action_shape}
    elif self.qd_agent == 'DMP':
      self.agent_shapes = {'dof': 2, 'degree': 3}
    # ---------------------------------------------------------

    # Metric
    # ---------------------------------------------------------
    self.gpu = True
    self.rnd_input = 'image'
    self.feature_size = 64
    self.learning_rate = 0.01
    # ---------------------------------------------------------

    # Save Path
    self.save_path = os.path.join(utils.get_projectpath(), 'experiments', self.exp_name)

  def _get_dict(self):
    params_dict = {key:value for key, value in self.__dict__items() if not key.startswith('__') and not callable(key)}
    return params_dict

  def save(self):
    if not os.path.exists(self.save_path):
      os.mkdir(self.save_path)
    with open(os.path.join(self.save_path, 'params.json'), 'w') as f:
      json.dump(self._get_dict(), f, indent=4)

