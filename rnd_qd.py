import numpy as np
from core.rnd import rnd
from core.qd import population, agents
from core import optimizer
import gym, torch
import multiprocessing as mp
import os
import gym_billiard
# env_tag = 'Billiard-v0'
env_tag = 'MountainCarContinuous-v0'


class RndQD(object):

  def __init__(self, env, action_shape, obs_shape, bs_shape, pop_size, use_novelty=True, use_archive=False, gpu=False):
    '''

    :param env: Environment in which we act
    :param action_shape: dimension of the action space
    :param obs_shape: dimension of the observation space
    :param bs_shape: dimension of the behavious space
    '''
    self.pop_size = pop_size
    self.use_novelty = use_novelty
    self.parameters = None
    self.env = env
    self.population = population.Population(agents.FFNeuralAgent,
                                            input_shape=obs_shape,
                                            output_shape=action_shape,
                                            pop_size=self.pop_size)
    self.archive = None
    if use_archive:
      self.archive = population.Population(agents.FFNeuralAgent,
                                           input_shape=obs_shape,
                                           output_shape=action_shape,
                                           pop_size=0)
    if gpu:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device('cpu')
    if self.use_novelty:
      self.metric = rnd.RND(input_shape=obs_shape, encoding_shape=bs_shape, pop_size=self.pop_size)
    else:
      self.metric = None
    self.opt = optimizer.NoveltyOptimizer(self.population, archive=self.archive)
    self.cumulated_state = []

  # TODO make this run in parallel
  def evaluate_agent(self, agent):
    '''
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    '''
    done = False
    cumulated_reward = 0

    obs = self.env.reset()
    if env_tag is 'Billiard-v0':
      obs = np.concatenate(obs)
    if self.use_novelty: state = np.array([obs])
    obs = np.array([obs])
    while not done:
      action = np.squeeze(agent['agent'](obs))

      obs, reward, done, info = self.env.step([action])
      if env_tag is 'Billiard-v0':
        obs = np.concatenate(obs)

      obs = np.array([obs])
      if self.use_novelty: state = np.append(state, obs, axis=0)

      cumulated_reward += reward

    surprise = 0
    if self.use_novelty:
      state = torch.Tensor(state)
      surprise = self.metric(state.unsqueeze(0))# Input Dimensions need to be [1, traj_len, obs_space]
      surprise = surprise.cpu().data.numpy()
      agent['bs'] = np.array([[obs[0][0], 0]])
      self.cumulated_state.append(state) # Append here all the states

    agent['surprise'] = surprise
    agent['reward'] = cumulated_reward

  def update_rnd(self):
    '''
    This function uses the cumulated state to update the rnd nets and then empties the cumulated_state
    :return:
    '''
    # Find max trajectory length
    max_t_len = 0
    for k in self.cumulated_state:
      if k.size()[0] > max_t_len:
        max_t_len = k.size()[0]
    # Pad trajectories
    for idx, k in enumerate(self.cumulated_state):
      while k.size()[0] < max_t_len:
        k = torch.cat((k, torch.zeros_like(k[:1])))
      self.cumulated_state[idx] = k

    self.cumulated_state = torch.stack(self.cumulated_state)
    cum_surprise = self.metric.training_step(self.cumulated_state)
    self.cumulated_state = []
    return cum_surprise

  def train(self, steps=10000):
    '''
    This function trains the agents and the RND
    :param steps: number of update steps (or generations)
    :return:
    '''
    self.elapsed_gen = 0
    for self.elapsed_gen in range(steps):
      cs = 0
      max_rew = -np.inf
      for a in self.population:
        self.evaluate_agent(a)
        cs += a['surprise']
        if max_rew < a['reward']:
          max_rew = a['reward']
      if self.use_novelty:
        self.update_rnd() # From here we can get the cumulated surprise on the same state as cs but after training

      self.opt.step()
      if self.elapsed_gen % 10 == 0:
        print('Generation {}'.format(self.elapsed_gen))
        if self.archive is not None:
          print('Archive size {}'.format(self.archive.size))
        print('Average generation surprise {}'.format(cs/self.pop_size))
        print('Max reward {}'.format(max_rew))
        # self.show()
        print()

  def save(self, filepath):
    if not os.path.exists(filepath):
      try:
        os.mkdir(os.path.abspath(filepath))
      except:
        print('Cannot create save folder.')
    self.population.save_pop(filepath, 'pop')
    self.archive.save_pop(filepath, 'archive')
    self.metric.save(filepath)

  def show(self):
    print('Behaviour space coverage representation.')
    if self.archive is not None:
      bs_points = np.concatenate(self.archive['bs'].values)
    else:
      bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
      print(bs_points)
    import matplotlib.pyplot as plt

    pts = ([x[0] for x in bs_points if x is not None], [y[1] for y in bs_points if y is not None])
    # plt.scatter(pts[0], pts[1])
    plt.hist(pts[0])
    plt.show()


if __name__ == '__main__':
  import time
  env = gym.make(env_tag)

  env.seed()
  np.random.seed()
  torch.initial_seed()

  rnd_qd = RndQD(env, action_shape=1, obs_shape=2, bs_shape=512, pop_size=50, use_novelty=True, use_archive=True, gpu=True)
  try:
    rnd_qd.train()
  except KeyboardInterrupt:
    print('User Interruption.')

  if rnd_qd.archive is None:
    pop = rnd_qd.population
  else:
    pop = rnd_qd.archive
  print('Total generations: {}'.format(rnd_qd.elapsed_gen))
  print('Archive length {}'.format(pop.size))

  # try:
  #   print('Testing best reward')
  #   obs = rnd_qd.env.reset()
  #
  #   rewards = rnd_qd.archive['reward'].sort_values(ascending=False)
  #   best = rnd_qd.archive[rewards.iloc[:1].index.values[0]]  # Get best
  #
  #   print('Best archive reward {}'.format(best['reward']))
  #   for _ in range(1000):
  #     rnd_qd.env.render()
  #     action = np.squeeze(best['agent'](np.array([obs])))
  #     # if action > 0:
  #     #   action = 1
  #     # else:
  #     #   action = 0
  #
  #     obs, reward, done, info = rnd_qd.env.step([action])
  #     if done:
  #       obs = rnd_qd.env.reset()
  # except KeyboardInterrupt:
  #   print('User Interruption.')

  # if rnd_qd.use_novelty:
  #   print('Testing best surprise')
  #   obs = rnd_qd.env.reset()
  #   surprises = rnd_qd.archive['surprise'].sort_values(ascending=False)
  #   best = rnd_qd.archive[surprises.iloc[:1].index.values[0]]  # Get best
  #
  #   print('Best archive surprise {}'.format(best['surprise']))
  #   for _ in range(1000):
  #     rnd_qd.env.render()
  #     action = np.squeeze(best['agent'](np.array([obs])))
  #     # if action > 0:
  #     #   action = 1
  #     # else:
  #     #   action = 0
  #
  #     obs, reward, done, info = rnd_qd.env.step([action])
  #     if done:
  #       obs = rnd_qd.env.reset()

  rnd_qd.show()

  print('Testing result according to best reward.')
  rewards = pop['reward'].sort_values(ascending=False)
  for idx in range(pop.size):
    tested = pop[rewards.iloc[idx:idx+1].index.values[0]]
    print()
    print('Testing agent with reward {}'.format(tested['reward']))
    done = False
    ts = 0
    obs = rnd_qd.env.reset()
    while not done and ts < 1000:
      if env_tag is 'Billiard-v0':
        obs = np.concatenate(obs)
      rnd_qd.env.render()
      action = np.squeeze(tested['agent'](np.array([obs])))
      obs, reward, done, info = rnd_qd.env.step([action])


