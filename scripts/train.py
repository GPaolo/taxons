# Created by Giuseppe Paolo 
# Date: 15/02/19

from core import rnd_qd
from baselines import novelty_search, policy_space, random_search, random_bd, image_bd
import gym, torch
import gym_billiard, gym_fastsim, pybulletgym
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
  env = gym.make(params.env_tag) # Create environment
  # Set seed
  params.seed = seed
  env.seed(seed)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  params.save()

  if not os.path.exists(params.save_path):
    os.mkdir(params.save_path)

  # Load method evolver
  if params.exp == 'NS':
    evolver = novelty_search.NoveltySearch(env=env, parameters=params)
  elif params.exp== 'PS':
    evolver = policy_space.PolicySpace(env=env, parameters=params)
  elif params.exp == 'RS':
    evolver = random_search.RandomSearch(env=env, parameters=params)
  elif params.exp == 'RBD':
    evolver = random_bd.RandomBD(env=env, parameters=params)
  elif params.exp == 'IBD':
    evolver = image_bd.ImageBD(env=env, parameters=params)
  else:
    evolver = rnd_qd.RndQD(env=env, parameters=params)

  # Start training
  start_time = time.monotonic()
  try:
    evolver.train(params.generations)
  except KeyboardInterrupt:
    print('Seed {} - User Interruption.'.format(seed))
  except Exception as e:
    print("Seed {} - EXCEPTION: {}".format(seed, traceback.format_exc()))
  end_time = time.monotonic()
  total_train_time += (end_time - start_time)

  # Save
  evolver.save()
  params.save()

  # Print some informations
  if evolver.archive is None:
    pop = evolver.population
  else:
    pop = evolver.archive
  print('Seed {} - Total generations: {}'.format(seed, evolver.elapsed_gen+1))
  print('Seed {} - Archive length {}'.format(seed, pop.size))
  print('Seed {} - Training time {}'.format(seed, timedelta(seconds=total_train_time)))

  # Plot coverage
  if evolver.archive is not None:
    bs_points = np.stack(evolver.archive['bs'].to_list())
  else:
    bs_points = np.concatenate([a['bs'] for a in evolver.population if a['bs'] is not None])
  if 'Ant' in params.env_tag:
    u_limit = 3.5
    l_limit = -u_limit
  elif 'FastsimSimpleNavigation' in params.env_tag:
    u_limit = 600
    l_limit = 0
  else:
    u_limit = 1.35
    l_limit = -u_limit

  utils.show(bs_points, filepath=params.save_path,
             name='final_{}_{}'.format(evolver.elapsed_gen, params.env_tag),
             info={'seed':seed},
             upper_limit=u_limit, lower_limit=l_limit)


if __name__ == "__main__":
  p = parameters.Params()
  parallel_threads = p.threads
  seeds = [11, 59,
          3, 6, 4,
          18, 13, 1,
          22, 34, 99,
          43, 100, 15,
          66, 10,7,
          9, 42, 2
          ]

  # Selects seeds to run in parallel
  multiseeds = []
  for i in range(0, len(seeds), parallel_threads):
    multiseeds.append(seeds[i: min(i+parallel_threads, len(seeds))])

  total_train_time = 0

  for seeds in multiseeds:
    params = [parameters.Params() for i in range(len(seeds))] # Get parameters for seed
    print('Experiment description:\n{}'.format(params[0].info))

    if params[0].parallel: # Run in parallel
      nodes = min(len(seeds), pathos.threading.cpu_count()-1) # Create threads
      print('Creating {} threads...'.format(nodes))
      with ProcessPool(nodes=nodes) as pool:
        start_time = time.monotonic()
        try:
          results = pool.map(main, seeds, params) # Pool main functions to threads
        except KeyboardInterrupt:
          break
        end_time = time.monotonic()
      total_train_time += (end_time - start_time)

    else: # Run sequentially
      end = False
      for seed, par in zip(seeds, params):
        start_time = time.monotonic()
        try:
          results = main(seed, par) # Launch main function
        except KeyboardInterrupt:
          end = True
        end_time = time.monotonic()
        total_train_time += (end_time - start_time)
        if end: break
    # Do garbage collect
    gc.collect()

  print('\nTotal training time: \n{}\n'.format(timedelta(seconds=total_train_time)))
