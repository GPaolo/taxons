import numpy as np
from core.rnd import rnd
from core.qd import population, agents
from core import optimizer
import gym, torch

env_tag = 'MountainCarContinuous-v0'

class RndQD(object):

  def __init__(self):
    self.env = gym.make(env_tag)
    self.population = population.Population(agents.NeuralAgent,
                                            input_shape=self.env.observation_space.shape[0],
                                            output_shape=self.env.action_space.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  pop = population.Population(agent=agents.NeuralAgent, output_shape=1, input_shape=4)
  metric = rnd.RND(input_shape=4, encoding_shape=3)

  opt = optimizer.SimpleOptimizer(env, pop, metric)
  for _ in range(10):
    opt.step()


