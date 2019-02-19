# Created by Giuseppe Paolo 
# Date: 15/02/19

from sacred import Experiment
from core import rnd_qd
import gym, torch
import gym_billiard
import numpy as np
from core.qd import agents
from core.utils import utils
from sacred.observers import FileStorageObserver
import os

ex = Experiment()

@ex.config
def config():
  info = 'Testing DMP based on 3rd degree polynomial. Here I mutate also the ones added to the archive'

  exp_name = 'total_mutation'
  seed = 7

  # Environment
  # ---------------------------------------------------------
  action_shape = 2
  env_tag = 'Billiard-v0'  # MountainCarContinuous-v0'
  # ---------------------------------------------------------

  # QD
  # ---------------------------------------------------------
  generations = 500
  pop_size = 100
  use_novelty = True
  use_archive = True

  qd_agent = 'DMP' #'DMP
  if qd_agent == 'Neural':
    agent_shapes = {'input_shape':6, 'output_shape':action_shape}
  elif qd_agent == 'DMP':
    agent_shapes = {'dof':2, 'degree':20}
  # ---------------------------------------------------------

  # RND
  # ---------------------------------------------------------
  rnd_input = 'image'
  rnd_output_size = 64
  # ---------------------------------------------------------

  # Save Path
  save_path = os.path.join(utils.get_projectpath(), 'experiments', exp_name)

  ex.observers.append(FileStorageObserver.create(save_path))


@ex.automain
def main(env_tag, seed,
         use_novelty, use_archive,
         pop_size, save_path,
         agent_shapes, rnd_output_size,
         generations, qd_agent):

  env = gym.make(env_tag)

  env.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  if not os.path.exists(save_path):
    os.mkdir(save_path)

  evolver = rnd_qd.RndQD(env=env,
                         agents_shapes=agent_shapes,
                         bs_shape=rnd_output_size,
                         pop_size=pop_size,
                         use_novelty=use_novelty,
                         use_archive=use_archive,
                         gpu=True,
                         save_path=save_path,
                         agent_name=qd_agent)
  try:
    evolver.train(generations)
  except KeyboardInterrupt:
    print('User Interruption.')

  evolver.save()

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
  utils.show(bs_points, filepath=save_path, name='final_{}_{}'.format(evolver.elapsed_gen, env_tag))

  print('Testing result according to best reward.')
  rewards = pop['reward'].sort_values(ascending=False)
  for idx in range(pop.size):
    tested = pop[rewards.iloc[idx:idx + 1].index.values[0]]
    print()
    print('Testing agent {} with reward {}'.format(tested['name'], tested['reward']))
    done = False
    ts = 0
    obs = utils.obs_formatting(env_tag, evolver.env.reset())
    while not done:
      evolver.env.render()

      if qd_agent == 'Neural':
        agent_input = obs
      elif qd_agent == 'DMP':
        agent_input = ts

      action = utils.action_formatting(env_tag, tested['agent'](agent_input))
      obs, reward, done, info = evolver.env.step(action)
      obs = utils.obs_formatting(env_tag, obs)
      ts += 1

