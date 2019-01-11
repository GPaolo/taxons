import numpy as np
from core.rnd import rnd
from core.qd import population, agents
from core import optimizer
import gym, torch
import multiprocessing as mp

env_tag = 'CartPole-v1'

class RndQD(object):

  def __init__(self):
    self.env = gym.make(env_tag)
    self.population = population.Population(agents.FFNeuralAgent,
                                            input_shape=self.env.observation_space.shape[0],
                                            output_shape=self.env.action_space.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#The evaluation of the agent is better outside of the optimizer. The optimizer should receive the pop already evaluated.
#This way we don't have to give the env to the optimizer and we are not tied to OpenAiGYm
#Also we can decide outside how to normalize the inputs, and what to give to the metric
def evaluate_agent(agent, env_tag, metric):
  # TODO have to find a way to run this in parallel
  env = gym.make(env_tag)

  done = False
  total_reward = 0
  total_surprise = 0

  obs = env.reset()
  while not done:
    action = np.squeeze(agent['agent'](np.array([obs])))
    if action > 0:
      action = 1
    else:
      action = 0

    obs, reward, done, info = env.step(action)
    state = torch.Tensor([obs])
    # TODO add whitening of inputs to the metric (see RND paper sec 2.4)
    surprise = metric.training_step(state)

    total_reward += reward
    total_surprise += surprise.cpu().item()

  agent['surprise'] = total_surprise
  agent['reward'] = total_reward

if __name__ == '__main__':
  pop = population.Population(agent=agents.FFNeuralAgent, output_shape=1, input_shape=4)
  metric = rnd.RND(input_shape=4, encoding_shape=3)
  env = gym.make(env_tag)

  opt = optimizer.SimpleOptimizer(pop)
  try:
    for _ in range(100000):
      if _ % 1000 == 0:
        print('Step {}'.format(_))
      for a in pop:
        evaluate_agent(a, env_tag, metric)
      opt.step()
  except KeyboardInterrupt:
    print('User interruption')


  print('Testing best reward')
  obs = env.reset()
  best = pop[0]
  for a in pop:
    if a['reward'] > best['reward']:
      best = a

  print('Best reward {}'.format(best['reward']))
  for _ in range(1000):
    env.render()
    action = np.squeeze(best['agent'](np.array([obs])))
    if action > 0:
      action = 1
    else:
      action = 0

    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()

  print('Testing best surprise')
  obs = env.reset()
  best = pop[0]
  for a in pop:
    if a['surprise'] > best['surprise']:
      best = a

  print('Best surprise {}'.format(best['surprise']))
  for _ in range(1000):
    env.render()
    action = np.squeeze(best['agent'](np.array([obs])))
    if action > 0:
      action = 1
    else:
      action = 0

    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()



