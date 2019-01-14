import numpy as np
from abc import ABCMeta, abstractmethod  # This is to force implementation of child class methods
import random

class BaseOptimizer(metaclass=ABCMeta):
  def __init__(self, pop, mutation_rate=.9, sync_update=True, archive=None):
    self.pop = pop
    self.mutation_rate = mutation_rate
    self.sync_update = sync_update
    self.archive = archive

  def _get_pareto_front(self, costs, direction='max'):
    '''
    This function calculates the agents in the pareto front. Is taken from:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    :param: costs: list of costs along which calculate the front. Shape [pop_len, num_of_pareto_dim]
    :param: direction: can be either 'min' or 'max'. Defines The way we calculate the front.
    :return: The boolean mask of the pareto efficient elements.
    '''
    assert direction in ['min', 'max'], 'Direction can be either min or max. Is {}'.format(direction)
    is_pareto = np.arange(len(costs))

    i = 0
    while i < len(costs):
      if direction == 'max':
        nondominated_point_mask = np.any(costs > costs[i], axis=1)
      elif direction == 'min':
        nondominated_point_mask = np.any(costs < costs[i], axis=1)
      nondominated_point_mask[i] = True
      is_pareto = is_pareto[nondominated_point_mask]
      costs = costs[nondominated_point_mask]
      i = np.sum(nondominated_point_mask[:i]) + 1

    return is_pareto

  @abstractmethod
  def step(self):
    '''
    Optimizer step
    '''
    pass


class SimpleOptimizer(BaseOptimizer):

  def step(self):
    """
    This function performs an optimization step.
    Once the agents have been evaluated, it calculates the pareto front of the agents and decides who and how is
    going to reproduce. It also mutates the agents.
    :return:
    """
    # Find best agents
    costs = np.array([np.array([a['surprise'], a['reward']]) for a in self.pop])
    is_pareto = self._get_pareto_front(costs)

    # Create new gen by substituting random agents with copies of the best ones. (Also the best ones can be subst, effectively
    # reducing the amount of dead agents)
    new_gen = [self.pop[i].copy() for i in is_pareto]

    for i in is_pareto:
      self.pop[i]['best'] = True
    dead = random.sample(range(self.pop.size), len(new_gen))
    for i, new_agent in zip(dead, new_gen):
      self.pop[i] = new_agent

    # Mutate pop that are not pareto optima
    for a in self.pop:
      if np.random.random() <= self.mutation_rate and not a['best']:
        a['agent'].mutate()










if __name__ == '__main__':
  a = [1,2,3,4,5,6,7,8,9,0]
  for k in np.random.randint(0, 10 + 1, 3):
    print(k)





