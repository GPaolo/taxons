import numpy as np


# ----------------------------------------------------------
class BaseOptimizer(object):
  """
  Bsse optimizer class
  """
  # -----------------------------
  def __init__(self, pop, mutation_rate=.9, archive=None, metric_update_interval=30):
    self.pop = pop
    self.archive = archive
    self.mutation_rate = mutation_rate
    self.step_count = 0
    self.min_surprise = 0
    self.metric_update_interval = metric_update_interval
  # -----------------------------

  # -----------------------------
  def measure_novelty(self):
    """
    This function calculates the novelty of each agent in the population using the features descriptor.
    The novelty is calculated wrt to the population and the archive.
    :return:
    """
    for agent_idx in range(self.pop.size):
      bs_point = self.pop[agent_idx]['features'][0] # Get agent features

      bs_space = np.stack([a[0] for a in self.pop['features'].values]) # Get pop features
      bs_space = np.delete(bs_space, agent_idx, axis=0) # Remove agent features from list
      if self.archive.size > 0:
        archive_bs_space = np.stack([a[0] for a in self.archive['features'].values]) # Get archive features
        bs_space = np.concatenate([bs_space, archive_bs_space]) # Stack pop and archive feats

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
  # -----------------------------

  # -----------------------------
  def update_archive_surprise(self):
    """
    This function updates the archive and the pop according to the surprise metric.
    """
    novel = self.pop['surprise'].sort_values(ascending=False)
    best = novel.iloc[:5].index.values # Get 5 best agents
    worst = novel.iloc[-len(best):].index.values  # Get worst ones

    if self.archive is not None:
      for idx in best:
        if self.pop[idx]['name'] not in self.archive['name'].values:
          self.archive.add(self.pop.copy(idx, with_data=True))  # Only add the most novel ones

    # Get a copy of the best agents
    new_gen = []
    for i in best:
      new_gen.append(self.pop.copy(i))
      self.pop[i]['best'] = True

    # Substitute worst agents with the copy of the best ones
    for i, new_agent in zip(worst, new_gen):
      self.pop[i] = new_agent
  # -----------------------------

  # -----------------------------
  def update_archive_novelty(self):
    """
    This function updates the archive adn the pop according to the novelty metric.
    """
    novel = self.pop['novelty'].sort_values(ascending=False)
    best = novel.iloc[:5].index.values  # Get 5 best
    dead = novel.iloc[-5:].index.values  # Get 5 worst

    if self.archive is not None:
      for idx in best:
        if self.pop[idx]['name'] not in self.archive['name'].values:
          self.archive.add(self.pop.copy(idx, with_data=True))  # Only add the most novel ones

    # Get a copy of the best ones
    new_gen = []
    for i in best:
      new_gen.append(self.pop.copy(i))
      self.pop[i]['best'] = True

    # Substitute worst agents with the copy of the best ones
    for i, new_agent in zip(dead, new_gen):
      self.pop[i] = new_agent
  # -----------------------------

  # -----------------------------
  def mutate_pop(self):
    """
    This function mutates the population
    """
    for a in self.pop:
      if np.random.random() <= self.mutation_rate:
        a['agent'].mutate()
        a['name'] = self.pop.agent_name  # When an agent is mutated it also changes name, otherwise it will never be added to the archive
        self.pop.agent_name += 1
      a['best'] = False

    self.step_count += 1
  # -----------------------------

  # -----------------------------
  def step(self, **kwargs):
    """
    Optimizer step. Needs to be implemented by inheriting classes
    """
    raise NotImplementedError
  # -----------------------------
# ----------------------------------------------------------


# ----------------------------------------------------------
class FitnessOptimizer(BaseOptimizer):
  """
  Optimizer that looks for the fitness of the agents
  """
  # -----------------------------
  def step(self, **kwargs):
    """
    Optimization step
    :param kwargs:
    """
    rewards = self.pop['reward'].sort_values(ascending=False) # Get rewards of population
    best = rewards.iloc[:5].index.values # Get 5 best
    worst = rewards.iloc[-5:].index.values # Get 5 worst

    # Get a copy of the best agents
    new_gen = []
    for i in best:
      new_gen.append(self.pop.copy(i))
      self.pop[i]['best'] = True

    # Substitute worst agents with the copy of the best ones
    for i, new_agent in zip(worst, new_gen):
      self.pop[i] = new_agent

    # Mutate pop that are not best
    for a in self.pop:
      if np.random.random() <= self.mutation_rate and not a['best']:
        a['agent'].mutate()
      a['best'] = False # For the new gen no one is best yet
  # -----------------------------
# ----------------------------------------------------------


# ----------------------------------------------------------
class NoveltySurpriseOptimizer(BaseOptimizer):
  """
  Optimizer that randomly selects between novelty and surprise metric for the agents
  """
  # -----------------------------
  def step(self, **kwargs):
    """
    Performs optimization step. It does it by updating archive and pop novelty, deciding which metric to use to update the archive
    and then mutates the agents.
    """
    self.measure_novelty() # Update novelty
    archive_len = len(self.archive)

    if self.step_count < 30:
      print('Using Novelty update')
      self.update_archive_novelty()
    else:
      if np.random.uniform() <= 0.5:
        print('Using Novelty update')
        self.update_archive_novelty()
      else:
        print('Using Surprise update')
        self.update_archive_surprise()
    print("Max surprise {}".format(np.max(self.pop['surprise'])))
    print('Added to archive: {}'.format(len(self.archive)-archive_len))

    self.mutate_pop()
  # -----------------------------
# ----------------------------------------------------------


# ----------------------------------------------------------
class SurpriseOptimizer(BaseOptimizer):
  """
  Optimizer that uses only the surprise as metric
  """
  def step(self, **kwargs):
    """
    This function performs an optimization step by taking the agents with the highest surprise. The surprise is the error
    of the network.
    """
    self.update_archive_surprise()
    self.mutate_pop()
# ----------------------------------------------------------


# ----------------------------------------------------------
class NoveltyOptimizer(BaseOptimizer):
  """
  Optimizer that uses only the novelty as metric
  """
  def step(self, **kwargs):
    """
    This function optimizes the population according to classic novelty metric.
    :param kwargs:
    :return:
    """
    self.measure_novelty()
    self.update_archive_novelty()

    self.mutate_pop()
# ----------------------------------------------------------



