import numpy as np
from abc import ABCMeta, abstractmethod  # This is to force implementation of child class methods
import random
from copy import deepcopy

class BaseOptimizer(metaclass=ABCMeta):
  def __init__(self, pop, mutation_rate=.9, archive=None):
    self.pop = pop
    self.archive = archive
    self.mutation_rate = mutation_rate

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
  def step(self, **kwargs):
    '''
    Optimizer step
    '''
    pass


class FitnessOptimizer(BaseOptimizer):

  def step(self, **kwargs):
    rewards = self.pop['reward'].sort_values(ascending=False)
    best = rewards.iloc[:5].index.values # Get 5 best
    worst = rewards.iloc[-5:].index.values # Get 5 worst

    new_gen = []
    for i in best:
      new_gen.append(self.pop.copy(i))
      self.pop[i]['best'] = True

    for i, new_agent in zip(worst, new_gen):
      self.pop[i] = new_agent

    # Mutate pop that are not best
    for a in self.pop:
      if np.random.random() <= self.mutation_rate and not a['best']:
        a['agent'].mutate()

# TODO
class NSGCOptimizer(BaseOptimizer):

  def step(self, **kwargs):
    '''
    Perform optimization step according to NSGC procedure. We add to the archive only according to novelty.
    :return:
    '''
    # Find best agents
    costs = np.array([np.array([a['surprise'], a['reward']]) for a in self.pop])
    is_pareto = self._get_pareto_front(costs)

    # If archive is empty, add the whole pop to it.
    if len(self.archive) == 0:
      for a in self.pop:
        self.archive.add(deepcopy(a))
    else: # Otherwise add the novel if
      for a in self.pop:
        closest = np.array([np.linalg.norm(a['bs'] - arch['bs']) for arch in self.archive]).argmin()
        if np.linalg.norm(a['bs'] - self.archive[closest]['bs']) > 0.1:
          self.archive.add(a)
        elif a['reward'] > self.archive[closest]['reward']:
          self.archive[closest] = a

    # Create new gen by substituting random agents with copies of the best ones.
    # (Also the best ones can be subst, effectively reducing the amount of dead agents)
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


class ParetoOptimizer(BaseOptimizer):

  def step(self, **kwargs):
    """
    Once the agents have been evaluated, it calculates the pareto front of the agents and decides who and how is
    going to reproduce. It also mutates the agents.
    :return:
    """
    # Find best agents
    costs = np.array([self.pop['reward'].values, self.pop['surprise'].values]).transpose()
    is_pareto = self._get_pareto_front(costs)

    if self.archive is not None:
      if self.archive.size == 0: # First step, so copy all the pareto
        for idx in is_pareto:
          self.archive.add(self.pop.copy(idx, with_data=True))
      else: # add only the non dominated
        arch_costs = np.array([np.stack(self.archive['surprise'].values), np.stack(self.archive['reward'].values)]).transpose()
        for idx in is_pareto:
          if self.pop[idx]['name'] not in self.archive['name'].values: # Add an element in the archive only if not present already
            costs = np.array([self.pop[idx]['surprise'], self.pop[idx]['reward']])
            if np.any(np.any(costs > arch_costs, axis=1)): # TODO Invece di fare cosi potrei ricalcolare il pareto front dell'archivio+quello da aggiungere e vedere se ci sta. Se ci sta lo aggiungo
              self.archive.add(self.pop.copy(idx, with_data=True))
              self.pop[idx]['best'] = True

    # TODO NB: un'altra cosa che potrei fare e' far riprodurre solo quelli che aggiungo all'archivio.

    # Create new gen by substituting random agents with copies of the best ones.
    # (Also the best ones can be subst, effectively reducing the amount of dead agents)
    new_gen = []
    for i in is_pareto:
      if self.pop[i]['best']:
        new_gen.append(self.pop.copy(i)) # Reproduce only if has been added to the archive. This way we push exploration


    dead = random.sample(range(self.pop.size), len(new_gen))
    for i, new_agent in zip(dead, new_gen):
      self.pop[i] = new_agent

    # Mutate pop that are not pareto optima
    for a in self.pop:
      if np.random.random() <= self.mutation_rate and not a['best']:
        a['agent'].mutate()
        a['name'] = self.pop.agent_name  # When an agent is mutated it also changes name, otherwise it will never be added to the archive
        self.pop.agent_name += 1
      a['best'] = False


# TODO change this to literature implementation
class NoveltyOptimizer(BaseOptimizer):
  def step(self, **kwargs):
    '''
    This function performs an optimization step by taking the most novel agents
    :return:
    '''
    # if self.archive is not None:
    #   if self.archive.size == 0: # First step, so copy pop randomly
    #     for idx in range(self.pop.size):
    #       random_addition = np.random.uniform() <= 0.005
    #       if random_addition:
    #         self.archive.add(self.pop.copy(idx, with_data=True))
    #         self.pop[idx]['best'] = True
    #   else:
    #     novel = np.stack(self.archive['surprise'].values)
    #     self.archive.avg_surprise = np.mean(novel)
    #     for idx in range(self.pop.size):
    #       if self.pop[idx]['name'] not in self.archive['name'].values:  # Add an element in the archive only if not present already
    #         random_addition = np.random.uniform() <= 0.005
    #         if self.pop[idx]['surprise'] >= self.archive.avg_surprise or random_addition:
    #           self.archive.add(self.pop.copy(idx, with_data=True)) # Only add the most novel ones
    #           if not random_addition:
    #             self.pop[idx]['best'] = True
    #           if random_addition:
    #             print('Randomly selecting agent for archive.')
    novel = self.pop['surprise'].sort_values(ascending=False)
    best = novel.iloc[:5].index.values  # Get 5 best
    worst = novel.iloc[-5:].index.values  # Get 5 worst
    if self.archive is not None:
      for idx in best:
        if self.pop[idx]['name'] not in self.archive['name'].values:
          self.archive.add(self.pop.copy(idx, with_data=True))  # Only add the most novel ones

    # Maybe I should add to new gen only the ones that are added to the archive? In order not to have repetitions?
    # No, it does not makes sense, like this is better, so if one is still novel after the first time it can still
    # create ''novel'' kids.
    new_gen = []
    for i in best:
      new_gen.append(self.pop.copy(i))
      self.pop[i]['best'] = True

    for i, new_agent in zip(worst, new_gen):
      self.pop[i] = new_agent

    # new_gen = np.random.randint(self.pop.size, size=int(self.pop.size/5)) # Randomly select 1/5 of the pop to reproduce
    # dead = np.random.randint(self.pop.size, size=int(self.pop.size/5)) # Randomly select 1/5 of the pop to die    # new_gen = [self.pop[i[0]].copy() for i in novel[:3]] # Get first 3 most novel agents
    # for ng, d in zip(new_gen, dead):
    #   self.pop[d] = self.pop.copy(ng)
    #
    # new_gen = []  # Select only the most novel to reproduce
    # for i, a in enumerate(self.pop):
    #   if a['best']:
    #     new_gen.append(self.pop.copy(i))
    #
    # dead = random.sample(range(self.pop.size), len(new_gen))
    # for i, new_agent in zip(dead, new_gen):
    #   self.pop[i] = new_agent

    # Mutate pops
    for a in self.pop:
      if np.random.random() <= self.mutation_rate and not a['best']:
        a['agent'].mutate()
        a['name'] = self.pop.agent_name  # When an agent is mutated it also changes name, otherwise it will never be added to the archive
        self.pop.agent_name += 1
      a['best'] = False












if __name__ == '__main__':
  a = [1,2,3,4,5,6,7,8,9,0]
  for k in np.random.randint(0, 10 + 1, 3):
    print(k)





