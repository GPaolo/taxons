# Created by giuseppe
# Date: 28/03/19

from scripts import parameters
import gym, torch
import gym_billiard
import matplotlib.pyplot as plt
import numpy as np
from core.metrics import ae, rnd
from core.qd import population, agents
from core.utils import utils
import os

if __name__ == "__main__":

  # Parameters
  # -----------------------------------------------
  load_path = '/home/giuseppe/src/rnd_qd/experiments/Ant_High_Features/11'

  params = parameters.Params()
  params.load(os.path.join(load_path, 'params.json'))

  env = gym.make(params.env_tag)
  env.reset()
  # -----------------------------------------------

  # Possible targets
  # -----------------------------------------------
  x = []
  if "Billiard" in params.env_tag:
    env.env.params.RANDOM_BALL_INIT_POSE = True
  elif "Ant" in params.env_tag:
    env.render()

  for k in range(21):
    env.reset()
    if "Ant" in params.env_tag:
      for step in range(300):
        env.step(env.action_space.sample())
        CoM = np.array([env.env.data.qpos[:2]])
        if np.any(np.abs(CoM) >= np.array([3, 3])):
          break
    tmp = env.render(mode='rgb_array')
    x.append(tmp)
  x = np.stack(x)

  fig, ax = plt.subplots(7, 3)
  k = 0
  for i in range(7):
    for j in range(3):
      a = x[k]
      ax[i, j].imshow(a)
      ax[i, j].set_title(k)
      k += 1
  plt.show()
  if "Billiard" in params.env_tag:
    env.env.params.RANDOM_BALL_INIT_POSE = False
  # -----------------------------------------------

  # Load metric
  # -----------------------------------------------
  print('Loading metric...')
  if params.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device('cpu')

  if params.metric == 'AE':
    selector = ae.AutoEncoder(device=device, encoding_shape=params.feature_size)
    selector.load(os.path.join(load_path, 'models/ckpt_ae.pth'))
  elif params.metric == 'RND':
    selector = rnd.RND(params.feature_size)
    selector.load(os.path.join(load_path, 'models/ckpt_rnd.pth'))
  else:
    raise ValueError('Wrong metric selected: {}'.format(params.metric))
  # -----------------------------------------------

  # Load archive
  # -----------------------------------------------
  print('Loading agents...')
  if params.qd_agent == 'Neural':
    agent_type = agents.FFNeuralAgent
  elif params.qd_agent == 'DMP':
    agent_type = agents.DMPAgent
  else:
    raise ValueError('Wrong agent type selected: {}'.format(params.qd_agent))

  pop = population.Population(agent=agent_type, pop_size=0, shapes=params.agent_shapes)
  pop.load_pop(os.path.join(load_path, 'models/qd_archive.pkl'))
  # -----------------------------------------------

  # Evaluate archive agents BS points
  # -----------------------------------------------
  if pop[0]['features'] is None:
    for i, agent in enumerate(pop):
      if i % 50 == 0:
        print('Evaluating agent {}'.format(i))
      done = False
      obs = utils.obs_formatting(params.env_tag, env.reset())
      t = 0
      while not done:
        agent_input = t
        action = utils.action_formatting(params.env_tag, agent['agent'](agent_input))

        obs, reward, done, info = env.step(action)
        obs = utils.obs_formatting(params.env_tag, obs)
        t += 1
        if "Ant" in params.env_tag:
          CoM = np.array([env.env.data.qpos[:2]])
          if np.any(np.abs(CoM) >= np.array([3, 3])):
            done = True

      state = env.render(mode='rgb_array')
      state = state/np.max((np.max(state), 1))
      state = selector.subsample(torch.Tensor(state).permute(2, 0, 1).unsqueeze(0).to(device))
      surprise, bs_point, y = selector(state)
      bs_point = bs_point.flatten().cpu().data.numpy()
      agent['features'] = [bs_point]
  # -----------------------------------------------

  # Testing
  # -----------------------------------------------
  print('Choose the target image:')
  print('- A negative value to cycle between all the available targets')
  print('- A number in range [0, 22]')
  x_image = int(input(' '))
  if x_image < 0:
    print('Testing all of the targets...')
    x_image = list(range(len(x)))
  else:
    print("Testing on target {}".format(x_image))
    x_image = [x_image]

  for target in x_image:
    # Get new target BS point
    goal = torch.Tensor(x[target]).permute(2, 0, 1).unsqueeze(0).to(device)
    surprise, bs_point, reconstr = selector(goal/torch.max(torch.Tensor(np.array([torch.max(goal), 1]))))
    bs_point = bs_point.flatten().cpu().data.numpy()
    print('Target point surprise {}'.format(surprise.cpu().data))
    print('Target bs point {}'.format(bs_point))

    # Get K closest agents
    # -----------------------------------------------
    bs_space = np.stack([a[0] for a in pop['features'].values])

    # Get distances
    diff = np.atleast_2d(bs_space - bs_point)
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    print('Min distance {}'.format(np.min(dists)))

    k = 1 # Testing on just closest agent
    if len(dists) <= k:  # Should never happen
      idx = list(range(len(dists)))
      k = len(idx)
    else:
      idx = np.argpartition(dists, k)  # Get 15 nearest neighs

    mean_k_dist = np.mean(dists[idx[:k]])
    print('Mean K distance {}'.format(mean_k_dist))

    selected = pop[idx[:k]]
    print("Selected agent {}".format(idx[:k]))
    # -----------------------------------------------

    # Testing agent
    # -----------------------------------------------
    done = False
    ts = 0
    obs = utils.obs_formatting(params.env_tag, env.reset())
    while not done:
      env.render()
      agent_input = ts
      action = utils.action_formatting(params.env_tag, selected.iloc[0]['agent'](agent_input))
      obs, reward, done, info = env.step(action)
      obs = utils.obs_formatting(params.env_tag, obs)
      ts += 1
      CoM = np.array([env.env.data.qpos[:2]])
      if np.any(np.abs(CoM) >= np.array([3, 3])):
        done =True

    state = env.render(mode='rgb_array')
    state = state/np.max((np.max(state), 1))
    fig, ax = plt.subplots(3)
    ax[0].imshow(state)
    ax[1].imshow(x[target])
    ax[2].imshow(reconstr.permute(0,2,3,1)[0].cpu().data)
    plt.show()
    # -----------------------------------------------
  # -----------------------------------------------



