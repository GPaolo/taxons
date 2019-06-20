import numpy as np
from abc import ABCMeta, abstractmethod  # This is to force implementation of child class methods
import random


class BaseOptimizer(metaclass=ABCMeta):
  def __init__(self, pop, mutation_rate=.9, archive=None, metric_update_interval=30):
    self.pop = pop
    self.archive = archive
    self.mutation_rate = mutation_rate
    self.step_count = 0
    self.min_surprise = 0
    self.metric_update_interval = metric_update_interval

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

  def update_archive_pareto(self):
    # Find best agents
    costs = np.array([self.pop['novelty'].values, self.pop['surprise'].values]).transpose()
    is_pareto = self._get_pareto_front(costs)

    if self.archive is not None: # If using archive
      if self.archive.size == 0: #If first time updating archive
        for idx in is_pareto:
          self.archive.add(self.pop.copy(idx, with_data=True))
          self.pop[idx]['best'] = True
      else:
        arch_costs = np.array([np.stack(self.archive['surprise'].values), np.stack(self.archive['novelty'].values)]).transpose()
        for idx in is_pareto:
          self.pop[idx]['best'] = True
          if self.pop[idx]['name'] not in self.archive['name'].values:
            costs = np.array([self.pop[idx]['surprise'], self.pop[idx]['novelty']])
            if np.any(np.any(costs > arch_costs, axis=1)):
              self.archive.add(self.pop.copy(idx, with_data=True))

    # Create new gen by substituting random agents with copies of the best ones.
    # (Also the best ones can be subst, effectively reducing the amount of dead agents)
    new_gen = []
    for i in is_pareto:
      new_gen.append(self.pop.copy(i))  # Reproduce only if is on the Pareto Front

    dead = random.sample(range(self.pop.size), len(new_gen))
    for i, new_agent in zip(dead, new_gen):
      self.pop[i] = new_agent
      self.pop[i]['best'] = False  # The new generation is not considered as being on the pareto front, so it can be mutated. While the ones on the front are not

  def update_archive_surprise(self):
    novel = self.pop['surprise'].sort_values(ascending=False)
    best = novel[novel >= self.min_surprise].index.values # Get the ones with high enough surprise
    worst = novel.iloc[-len(best):].index.values  # Get worst ones
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

  def update_archive_novelty(self):
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
    Once the agents have been evaluated, it calculates the pareto front of the agents and decides who and how is
    going to reproduce. It also mutates the agents.
    :return:
    """
    self.measure_novelty()
    archive_len = len(self.archive)

    if self.step_count < 30:
      print('Using Novelty update')
      self.update_archive_novelty()
    else:
      if np.random.uniform() <= 0.5:
        print('Using Novelty update')
        self.update_archive_novelty()
      else:
        print('Using Surprise update') # TODO dai un'occhiata alla distr della surprise
        self.update_archive_surprise()
    self.min_surprise = np.max(self.pop['surprise'])# + 2*np.std(self.pop['surprise'])
    print("Max surprise {}".format(np.max(self.pop['surprise'])))
    print('Added to archive: {}'.format(len(self.archive)-archive_len))

    # Mutate pop
    for a in self.pop:
      if np.random.random() <= self.mutation_rate:
        a['agent'].mutate()
        a['name'] = self.pop.agent_name  # When an agent is mutated it also changes name, otherwise it will never be added to the archive
        self.pop.agent_name += 1
      a['best'] = False

    self.step_count += 1


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





