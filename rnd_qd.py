import numpy as np
from core.rnd import rnd
from core.qd import population, agents
from core import optimizer
import gym, torch
import multiprocessing as mp

env_tag = 'CartPole-v1'

class RndQD(object):

  def __init__(self, env, action_shape, obs_shape, bs_shape, pop_size, use_novelty=True):
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
    self.archive = population.Population(agents.FFNeuralAgent,
                                         input_shape=obs_shape,
                                         output_shape=action_shape,
                                         pop_size=0)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.use_novelty:
      self.metric = rnd.RND(input_shape=self.env.observation_space.shape[0], encoding_shape=bs_shape, pop_size=self.pop_size)
    else:
      self.metric = None
    self.opt = optimizer.ParetoOptimizer(self.population, archive=self.archive)
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
    if self.use_novelty: state = np.array([obs])
    obs = np.array([obs])
    while not done:
      action = np.squeeze(agent['agent'](obs))
      if action > 0:
        action = 1
      else:
        action = 0

      obs, reward, done, info = self.env.step(action)
      obs = np.array([obs])
      if self.use_novelty: state = np.append(state, obs, axis=0)

      cumulated_reward += reward

    surprise = 0
    if self.use_novelty:
      state = torch.Tensor(state)
      surprise = self.metric(state.unsqueeze(0)).cpu().data.numpy() # Input Dimensions need to be [1, traj_len, obs_space]
      bs_point = self.metric.get_bs_point(state.unsqueeze(0))
      agent['bs'] = bs_point.cpu().data.numpy()
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

  def train(self, steps=50000):
    '''
    This function trains the agents and the RND
    :return:
    '''
    self.elapsed_gen = 0
    for self.elapsed_gen in range(steps):
      cs = 0
      max_rew = 0
      for a in self.population:
        self.evaluate_agent(a)
        cs += a['surprise']
        if max_rew < a['reward']:
          max_rew = a['reward']
      if self.use_novelty:
        self.update_rnd() # From here we can get the cumulated surprise on the same state as cs but after training

      self.opt.step()
      if self.elapsed_gen % 1 == 0:
        print('Generation {}'.format(self.elapsed_gen))
        print('Average surprise {}'.format(cs/self.pop_size))
        print('Max reward {}'.format(max_rew))
        print()


if __name__ == '__main__':
  import time
  env = gym.make(env_tag)

  env.seed()
  np.random.seed()
  torch.initial_seed()

  rnd_qd = RndQD(env, action_shape=1, obs_shape=4, bs_shape=512, pop_size=25, use_novelty=True)
  try:
    rnd_qd.train()
  except KeyboardInterrupt:
    print('User Interruption.')
    print('Total generations: {}'.format(rnd_qd.elapsed_gen))

  try:
    print('Testing best reward')
    obs = rnd_qd.env.reset()

    rewards = rnd_qd.population['reward'].sort_values(ascending=False)
    best = rnd_qd.population[rewards.iloc[:1].index.values[0]]  # Get best

    print('Best reward {}'.format(best['reward']))
    for _ in range(1000):
      rnd_qd.env.render()
      action = np.squeeze(best['agent'](np.array([obs])))
      if action > 0:
        action = 1
      else:
        action = 0

      obs, reward, done, info = rnd_qd.env.step(action)
      if done:
        obs = rnd_qd.env.reset()
  except KeyboardInterrupt:
    print('User Interruption.')

  if rnd_qd.use_novelty:
    print('Testing best surprise')
    obs = rnd_qd.env.reset()
    surprises = rnd_qd.population['surprise'].sort_values(ascending=False)
    best = rnd_qd.population[surprises.iloc[:1].index.values[0]]  # Get best

    print('Best surprise {}'.format(best['surprise']))
    for _ in range(1000):
      rnd_qd.env.render()
      action = np.squeeze(best['agent'](np.array([obs])))
      if action > 0:
        action = 1
      else:
        action = 0

      obs, reward, done, info = rnd_qd.env.step(action)
      if done:
        obs = rnd_qd.env.reset()



