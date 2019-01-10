import numpy as np
from abc import ABCMeta, abstractmethod  # This is to force implementation of child class methods
from copy import deepcopy
import random

class BaseOptimizer(object):
  def __init__(self, env, pop, metric, mutation_rate=.9, sync_update=True):
    self.env = env
    self.pop = pop
    self.metric = metric
    self.mutation_rate = mutation_rate
    self.sync_update = sync_update

  def _get_pareto_front(self, costs):
    '''
    This function calculates the agents in the pareto front. Is taken from:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    :param: costs: list of costs along which calculate the front. Shape [pop_len, num_of_pareto_dim]
    :return: The boolean mask of the pareto efficient elements.
    '''
    is_pareto_mask = np.zeros(len(costs), dtype=bool)
    is_pareto = np.arange(len(costs))

    i = 0
    while i < len(costs):
      nondominated_point_mask = np.any(costs < costs[i], axis=1)
      nondominated_point_mask[i] = True
      is_pareto = is_pareto[nondominated_point_mask]
      costs = costs[nondominated_point_mask]
      i = np.sum(nondominated_point_mask[:i]) + 1

    is_pareto_mask[is_pareto] = True
    return is_pareto_mask

  @abstractmethod
  def evaluate_agent(self, agent):
    '''
    Function to evaluate single agent
    :param agent:
    :return:
    '''
    pass

  @abstractmethod
  def step(self):
    '''
    Optimizer step
    '''
    pass


class SimpleOptimizer(BaseOptimizer):

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
    Then it finds the best ones, copies them in random places of the pop and mutates all the pop.
    :return:
    """
    for agent in self.pop:
      self.evaluate_agent(agent)

    # Find best agents
    costs = np.array([np.array([a['surprise'], a['reward']]) for a in self.pop])
    pareto_mask = self._get_pareto_front(costs)

    # Create new gen by substituting random agents with copies of the best ones. (Also the best ones can be subst, effectively
    # reducing the amount of dead agents)
    new_gen = deepcopy(self.pop[pareto_mask])
    for a in self.pop[pareto_mask]:
      a['best'] = True
    dead = random.sample(range(self.pop.size+1), len(new_gen))
    for i, new_agent in zip(dead, new_gen):
      self.pop[i] = new_agent

    # Mutate pop that are not pareto optima
    for a in self.pop:
      if np.random.random() <= self.mutation_rate and not a['best']:
        a.mutate()










if __name__ == '__main__':
  a = [1,2,3,4,5,6,7,8,9,0]
  for k in np.random.randint(0, 10 + 1, 3):
    print(k)





