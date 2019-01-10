import numpy as np

class Optimizer(object):
  def __init__(self, env, pop, metric):
    self.env = env
    self.pop = pop
    self.metric = metric

  def evaluate_agent(self, agent):
    done = False
    total_reward = 0
    total_surprise =  0

    obs = self.env.reset()
    while not done:
      action = agent['agent'](obs)
      obs, reward, done, info = self.env.step(action)
      surprise = self.metric(obs)

      total_reward += reward
      total_surprise += surprise

    agent['surprise'] = total_surprise
    agent['reward'] = total_reward

  def step(self):
    """
    This function performs an optimization step. This consist in generating the new generation of the population.
    It does so by, for each member of the population, running a simulation in the env, and then assigning a value to it.
    :return:
    """
    for agent in self.pop:
      self.evaluate_agent(agent)


