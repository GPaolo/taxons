# Created by Giuseppe Paolo 
# Date: 15/02/19

from core import rnd_qd
import gym, torch
import gym_billiard
import numpy as np
from core.utils import utils, optimizer
import os
import json


class Params(object):
  def __init__(self):
    self.info = 'AE whose feature space dimension is really small.'

    self.exp_name = 'test_improvements'
    # Save Path
    self.save_path = os.path.join(utils.get_projectpath(), 'experiments', self.exp_name)
    self.seed = 7

    # Environment
    # ---------------------------------------------------------
    self.action_shape = 2
    self.env_tag = 'Billiard-v0'  # MountainCarContinuous-v0'
    # ---------------------------------------------------------

    # QD
    # ---------------------------------------------------------
    self.generations = 10
    self.pop_size = 100
    self.use_archive = True

    self.qd_agent = 'DMP'  # 'DMP
    if self.qd_agent == 'Neural':
      self.agent_shapes = {'input_shape': 6, 'output_shape': self.action_shape}
    elif self.qd_agent == 'DMP':
      self.agent_shapes = {'dof': 2, 'degree': 5}
    # ---------------------------------------------------------

    # Metric
    # ---------------------------------------------------------
    self.gpu = True
    self.metric = 'AE'  # 'RND'
    self.feature_size = 32
    self.learning_rate = 0.0001 # 0.0001 for RND
    self.per_agent_update = False
    self.state_recording_interval = 5
    self.max_states_recorded = 20
    self.update_metric = True
    # ---------------------------------------------------------

    # Optimizer
    # ---------------------------------------------------------
    self.optimizer_type = 'Pareto' # 'Surprise', 'Pareto'

    if self.optimizer_type == 'Novelty':
      self.optimizer = optimizer.NoveltyOptimizer
    elif self.optimizer_type == 'Surprise':
      self.optimizer = optimizer.SurpriseOptimizer
    elif self.optimizer_type == 'Pareto':
      self.optimizer = optimizer.ParetoOptimizer
    # ---------------------------------------------------------
  # -----------------------------------------Setup----------------

  def _get_dict(self):
    params_dict = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
    del params_dict['optimizer']
    params_dict['seed'] = params_dict['_seed']
    del params_dict['_seed']
    return params_dict
  # ---------------------------------------------------------

  def save(self):
    # if not os.path.exists(self.save_path):
    os.makedirs(self.save_path, exist_ok=True)
    with open(os.path.join(self.save_path, 'params.json'), 'w') as f:
      json.dump(self._get_dict(), f, indent=4)
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


if __name__ == "__main__":
  # seeds = [10, 7, 9, 42, 2]
  seeds = [10, 7]

  for seed in seeds:
    print('Training with seed {}'.format(seed))

    params = Params()
    env = gym.make(params.env_tag)

    params.seed = seed
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    params.save()

    if not os.path.exists(params.save_path):
      os.mkdir(params.save_path)

    evolver = rnd_qd.RndQD(env=env, parameters=params)
    try:
      evolver.train(params.generations)
    except KeyboardInterrupt:
      print('User Interruption.')

    evolver.save()
    params.save()

    if evolver.archive is None:
      pop = evolver.population
    else:
      pop = evolver.archive
    print('Total generations: {}'.format(evolver.elapsed_gen))
    print('Archive length {}'.format(pop.size))

    if evolver.archive is not None:
      bs_points = np.concatenate(evolver.archive['bs'].values)
    else:
      bs_points = np.concatenate([a['bs'] for a in evolver.population if a['bs'] is not None])
    utils.show(bs_points, filepath=params.save_path, name='final_{}_{}'.format(evolver.elapsed_gen, params.env_tag))

  # print('Testing result according to best reward.')
  # rewards = pop['reward'].sort_values(ascending=False)
  # for idx in range(pop.size):
  #   tested = pop[rewards.iloc[idx:idx + 1].index.values[0]]
  #   print()
  #   print('Testing agent {} with reward {}'.format(tested['name'], tested['reward']))
  #   done = False
  #   ts = 0
  #   obs = utils.obs_formatting(params.env_tag, evolver.env.reset())
  #   while not done:
  #     evolver.env.render()
  #
  #     if params.qd_agent == 'Neural':
  #       agent_input = obs
  #     elif params.qd_agent == 'DMP':
  #       agent_input = ts
  #
  #     action = utils.action_formatting(params.env_tag, tested['agent'](agent_input))
  #     obs, reward, done, info = evolver.env.step(action)
  #     obs = utils.obs_formatting(params.env_tag, obs)
  #     ts += 1

