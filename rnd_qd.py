import numpy as np
from core import rnd
from core.qd import population, agents
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
  # env = gym.make(env_tag)
  # print(env.action_space.shape[0])
  main = RndQD()
  print(main.env.action_space)

