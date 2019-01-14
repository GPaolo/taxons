import numpy as np
from core.rnd import rnd
from core.qd import population, agents
from core import optimizer
import gym, torch
import multiprocessing as mp

env_tag = 'CartPole-v1'

class RndQD(object):

  def __init__(self, env, action_shape, obs_shape):
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

    self.metric = rnd.RND(input_shape=self.env.observation_space.shape[0], encoding_shape=8)
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
    # TODO add whitening of inputs to the metric (see RND paper sec 2.4)
    surprise = self.metric(state)
    agent['surprise'] = surprise.cpu().item()
    agent['reward'] = cumulated_reward
    self.cumulated_state.append(state)

  def update_rnd(self):
    '''
    This function uses the cumulated state to update the rnd nets and then empties the cumulated_state
    :return:
    '''
    self.cumulated_state = torch.cat(self.cumulated_state)
    cum_surprise = self.metric.training_step(self.cumulated_state)
    self.cumulated_state = []
    return cum_surprise

  def train(self):
    '''
    This function trains the agents and the RND
    :return:
    '''
    for i in range(50000):
      for a in self.population:
        self.evaluate_agent(a)

      cs = self.update_rnd()
      if i % 1000 == 0:
        print('Generation {}'.format(i))
        print('Cumulated surprise {}'.format(cs))
        print()
      self.opt.step()


if __name__ == '__main__':
  env = gym.make(env_tag)
  main = RndQD(env, action_shape=1, obs_shape=4)
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



