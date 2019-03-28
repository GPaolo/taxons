# Created by Giuseppe Paolo 
# Date: 15/02/19

from core import rnd_qd
import gym, torch
import gym_billiard
import numpy as np
from core.utils import utils
import os
from scripts import parameters


if __name__ == "__main__":
  seeds = [10, 7, 9, 42, 2]

  for seed in seeds:
    print('Training with seed {}'.format(seed))

    params = parameters.Params()
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

