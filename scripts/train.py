# Created by Giuseppe Paolo 
# Date: 15/02/19

from core import rnd_qd
from baselines import novelty_search
import gym, torch
import gym_billiard
import numpy as np
from core.utils import utils
import os
from scripts import parameters
import time
from datetime import timedelta
import pathos
from pathos.pools import ProcessPool
import traceback
import gc
torch.backends.cudnn.enabled = False # There is a issue with CUDNN and Pytorch https://bit.ly/2ReLSDq


def main(seed, params):
  print('\nTraining with seed {}'.format(seed))
  total_train_time = 0
  env = gym.make(params.env_tag)
  params.seed = seed
  env.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  params.save()

  if not os.path.exists(params.save_path):
    os.mkdir(params.save_path)

  if params.baseline:
    evolver = novelty_search.NoveltySearch(env=env, parameters=params)
  else:
    evolver = rnd_qd.RndQD(env=env, parameters=params)

  start_time = time.monotonic()
  try:
    evolver.train(params.generations)
  except KeyboardInterrupt:
    print('Seed {} - User Interruption.'.format(seed))
  except Exception as e:
    print("Seed {} - EXCEPTION: {}".format(seed, traceback.format_exc()))

  end_time = time.monotonic()
  total_train_time += (end_time - start_time)

  evolver.save()
  params.save()

  if evolver.archive is None:
    pop = evolver.population
  else:
    pop = evolver.archive
  print('Seed {} - Total generations: {}'.format(seed, evolver.elapsed_gen+1))
  print('Seed {} - Archive length {}'.format(seed, pop.size))
  print('Seed {} - Training time {}'.format(seed, timedelta(seconds=total_train_time)))

  if evolver.archive is not None:
    bs_points = np.concatenate(evolver.archive['bs'].values)
  else:
    bs_points = np.concatenate([a['bs'] for a in evolver.population if a['bs'] is not None])
  if 'Ant' in params.env_tag:
    limit = 5
  else:
    limit = 1.35
  utils.show(bs_points, filepath=params.save_path,
             name='final_{}_{}'.format(evolver.elapsed_gen, params.env_tag),
             info={'seed':seed},
             limit=limit)

if __name__ == "__main__":
  parallel_threads = 6
  seeds = [11, 59, 3, 6, 4, 18, 13, 1, 22, 34, 99, 43, 100, 15, 66, 10, 7, 9, 42, 2]
  # seeds = [[7]]

  multiseeds = []
  for i in range(0, len(seeds), parallel_threads):
    multiseeds.append(seeds[i: min(i+parallel_threads, len(seeds))])

  total_train_time = 0

  for seeds in multiseeds:
    params = [parameters.Params() for i in range(len(seeds))]
    print('Experiment description:\n{}'.format(params[0].info))
    if params[0].parallel:
      nodes = min(len(seeds), pathos.threading.cpu_count()-1)
      print('Creating {} threads...'.format(nodes))
      # pool = ProcessPool(nodes=nodes)
      with ProcessPool(nodes=nodes) as pool:
        start_time = time.monotonic()
        try:
          results = pool.map(main, seeds, params)
        except KeyboardInterrupt:
          pass
        end_time = time.monotonic()
      total_train_time += (end_time - start_time)
      # pool.terminate()
      # pool.join()
    else:
      end = False
      for seed, par in zip(seeds, params):
        start_time = time.monotonic()
        try:
          results = main(seed, par)
        except KeyboardInterrupt:
          end = True
        end_time = time.monotonic()
        total_train_time += (end_time - start_time)
        if end: break
    gc.collect()
  # for res in results:
  #   utils.show(res[0], res[1], res[2])

  print('\nTotal training time: \n{}\n'.format(timedelta(seconds=total_train_time)))

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

