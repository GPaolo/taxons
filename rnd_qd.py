import numpy as np
from core.rnd import rnd
from core.qd import population, agents
from core import optimizer
import gym, torch
import multiprocessing as mp

env_tag = 'CartPole-v1'

class RndQD(object):

  def __init__(self, env, action_shape, obs_shape, bs_shape):
    '''

    :param env: Environment in which we act
    :param action_shape: dimension of the action space
    :param obs_shape: dimension of the observation space
    :param bs_shape: dimension of the behavious space
    '''
    self.parameters = None
    self.env = env
    self.population = population.Population(agents.FFNeuralAgent,
                                            input_shape=obs_shape,
                                            output_shape=action_shape)
    self.archive = population.Population(agents.FFNeuralAgent,
                                         input_shape=obs_shape,
                                         output_shape=action_shape,
                                         pop_size=0)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.metric = rnd.RND(input_shape=self.env.observation_space.shape[0], encoding_shape=bs_shape, pop_size=10)
    self.opt = optimizer.NoveltyOptimizer(self.population)
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
    state = np.array([obs])
    obs = np.array([obs])
    while not done:
      action = np.squeeze(agent['agent'](obs))
      if action > 0:
        action = 1
      else:
        action = 0

      obs, reward, done, info = self.env.step(action)
      obs = np.array([obs])
      state = np.append(state, obs, axis=0)

      cumulated_reward += reward

    state = torch.Tensor(state)
    surprise = self.metric(state.unsqueeze(0)) # Input Dimensions need to be [1, traj_len, obs_space]
    agent['surprise'] = surprise[0]
    agent['reward'] = cumulated_reward
    self.cumulated_state.append(state) # Append here all the states

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
    for i in range(steps):
      cs = 0
      for a in self.population:
        self.evaluate_agent(a)
        cs += a['surprise']
      self.update_rnd()
      self.opt.step()
      if i % 1000 == 0:
        print('Generation {}'.format(i))
        print('Cumulated surprise {}'.format(cs/10))
        print()


if __name__ == '__main__':
  env = gym.make(env_tag)
  main = RndQD(env, action_shape=1, obs_shape=4, bs_shape=2)
  try:
    main.train()
  except KeyboardInterrupt:
    print('User Interruption.')
  main.archive = main.population


  print('Testing best reward')
  obs = main.env.reset()
  best = main.archive[0]
  for a in main.archive:
    if a['reward'] > best['reward']:
      best = a

  print('Best reward {}'.format(best['reward']))
  for _ in range(3000):
    main.env.render()
    action = np.squeeze(best['agent'](np.array([obs])))
    if action > 0:
      action = 1
    else:
      action = 0

    obs, reward, done, info = main.env.step(action)
    if done:
      obs = main.env.reset()

  print('Testing best surprise')
  obs = main.env.reset()
  best = main.archive[0]
  for a in main.archive:
    if a['surprise'] > best['surprise']:
      best = a

  print('Best surprise {}'.format(best['surprise']))
  for _ in range(3000):
    main.env.render()
    action = np.squeeze(best['agent'](np.array([obs])))
    if action > 0:
      action = 1
    else:
      action = 0

    obs, reward, done, info = main.env.step(action)
    if done:
      obs = main.env.reset()



