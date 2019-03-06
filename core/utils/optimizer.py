import numpy as np
from abc import ABCMeta, abstractmethod  # This is to force implementation of child class methods
import random


class BaseOptimizer(metaclass=ABCMeta):
  def __init__(self, pop, mutation_rate=.9, archive=None):
    self.pop = pop
    self.archive = archive
    self.mutation_rate = mutation_rate

  def _get_pareto_front(self, costs, direction='max'):
    """
    This function calculates the agents in the pareto front. Is taken from:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    :param: costs: list of costs along which calculate the front. Shape [pop_len, num_of_pareto_dim]
    :param: direction: can be either 'min' or 'max'. Defines The way we calculate the front.
    :return: The boolean mask of the pareto efficient elements.
    """
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
    """
    Optimizer step
    """
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


class ParetoOptimizer(BaseOptimizer):

  def measure_novelty(self):
    """
    This function calculates the novelty of each agent in the population using the bs_point descriptor.
     The novelty is calculated wrt to the population and the archive.
    :return:
    """
    # MEASURE AGENT NOVELTY
    for agent_idx in range(self.pop.size):
      bs_point = self.pop[agent_idx]['features']

      bs_space = np.stack(self.pop['features'].values)
      bs_space = np.delete(bs_space, agent_idx, axis=0)
      if self.archive.size > 0:
        archive_bs_space = np.stack(self.archive['features'].values)
        bs_space = np.concatenate([bs_space, archive_bs_space])
      # Get distances
      diff = np.atleast_2d(bs_space - bs_point)
      dists = np.sqrt(np.sum(diff * diff, axis=1))
      k = 15
      if len(dists) <= k:  # Should never happen
        idx = list(range(len(dists)))
        k = len(idx)
      else:
        idx = np.argpartition(dists, k)  # Get 15 nearest neighs

      mean_k_dist = np.mean(dists[idx[:k]])
      self.pop[agent_idx]['novelty'] = mean_k_dist


  def step(self, **kwargs):
    """
    Once the agents have been evaluated, it calculates the pareto front of the agents and decides who and how is
    going to reproduce. It also mutates the agents.
    :return:
    """
    self.measure_novelty()

    # Find best agents
    costs = np.array([self.pop['novelty'].values, self.pop['surprise'].values]).transpose()
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


class SurpriseOptimizer(BaseOptimizer):
  def step(self, **kwargs):
    """
    This function performs an optimization step by taking the agent with the highest surprise. The surprise is the error
    of the network.
    :return:
    """
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

    # Mutate pops
    for a in self.pop:
      if np.random.random() <= self.mutation_rate:
        a['agent'].mutate()
        a['name'] = self.pop.agent_name  # When an agent is mutated it also changes name, otherwise it will never be added to the archive
        self.pop.agent_name += 1
      a['best'] = False


class NoveltyOptimizer(BaseOptimizer):

  def measure_novelty(self):
    """
    This function calculates the novelty of each agent in the population using the bs_point descriptor.
     The novelty is calculated wrt to the population and the archive.
    :return:
    """
    # MEASURE AGENT NOVELTY
    for agent_idx in range(self.pop.size):
      bs_point = self.pop[agent_idx]['features'][0]

      bs_space = np.stack([a[0] for a in self.pop['features'].values])
      bs_space = np.delete(bs_space, agent_idx, axis=0)
      if self.archive.size > 0:
        archive_bs_space = np.stack([a[0] for a in self.archive['features'].values])
        bs_space = np.concatenate([bs_space, archive_bs_space])
      # Get distances
      diff = np.atleast_2d(bs_space - bs_point)
      dists = np.sqrt(np.sum(diff * diff, axis=1))
      k = 15
      if len(dists) <= k:  # Should never happen
        idx = list(range(len(dists)))
        k = len(idx)
      else:
        idx = np.argpartition(dists, k)  # Get 15 nearest neighs

      mean_k_dist = np.mean(dists[idx[:k]])
      self.pop[agent_idx]['novelty'] = mean_k_dist

  def update_archive(self):
    """
    This function adds agents to the archive according to the novelty of each of them
    :return:
    """
    new_gen = []  # Reproduce only the novel ones
    # ADD AGENT TO ARCHIVE
    novel = self.pop['novelty'].sort_values(ascending=False)
    best = novel.iloc[:5].index.values  # Get 5 best
    dead = novel.iloc[-5:].index.values  # Get 5 worst
    if self.archive is not None:
      for idx in best:
        if self.pop[idx]['name'] not in self.archive['name'].values:
          self.archive.add(self.pop.copy(idx, with_data=True))  # Only add the most novel ones
    for i in best:
      new_gen.append(self.pop.copy(i))
      self.pop[i]['best'] = True

    # This one is common in both adaptive and non adaptive distance
    for i, new_agent in zip(dead, new_gen):
      self.pop[i] = new_agent

  def step(self, **kwargs):
    """
    This function optimizes the population according to classic novelty metric.
    :param kwargs:
    :return:
    """
    self.measure_novelty()
    self.update_archive()

    # Mutate pop
    for a in self.pop:
      if np.random.random() <= self.mutation_rate:
        a['agent'].mutate()
        a['name'] = self.pop.agent_name  # When an agent is mutated it also changes name, otherwise it will never be added to the archive
        self.pop.agent_name += 1
      a['best'] = False








if __name__ == '__main__':
  a = [1,2,3,4,5,6,7,8,9,0]
  for k in np.random.randint(0, 10 + 1, 3):
    print(k)





