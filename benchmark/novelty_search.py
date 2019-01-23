import numpy as np
from core.rnd import rnd
from core.qd import population, agents
from core import optimizer
import gym
import multiprocessing as mp
import os

env_tag = 'MountainCarContinuous-v0'

class NoveltySearch(object):
  def __init__(self, env):
    self.pop = population.Population(agent=agents.FFNeuralAgent, pop_size=25)
    self.archive = population.Population(agent=agents.FFNeuralAgent, pop_size=0)
    self.env = env

  def metric(self, agent):
    for a in self.archive:



if __name__ == '__main__':
  env = gym.make(env_tag)

  env.seed()
  np.random.seed()


