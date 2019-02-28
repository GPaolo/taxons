import numpy as np
from core.qd import population, agents
from core.utils import utils
import gym
import random
import threading
import gym_billiard
import matplotlib
import matplotlib.pyplot as plt
import os
# env_tag = 'MountainCarContinuous-v0'
env_tag = 'Billiard-v0'


class NoveltySearch(object):
  def __init__(self, env, filepath, obs_shape=6, action_shape=2, pop_size=50):
    self.obs_space = obs_shape
    self.max_arch_len = 100000
    self.action_space = action_shape
    self.pop = population.Population(agent=agents.DMPAgent,
                                     shapes={'dof':2, 'degree':3},
                                     pop_size=pop_size)
    self.archive = population.Population(agent=agents.DMPAgent,
                                         shapes={'dof':2, 'degree':3},
                                         pop_size=0)
    self.env = env
    self.min_dist = 0.5
    self.novel_in_gen = 0
    self.not_added = 0
    self.mutation_rate = 0.5
    self.thread = threading.Thread(target=self._show_progress)
    self.thread.start()
    self.adaptive_distance = False
    self.filepath = filepath
    if not os.path.exists(self.filepath):
      os.mkdir(self.filepath)

  def _show_progress(self):
    matplotlib.use('agg')
    print('If you want to show the progress, press s.')
    while True:
      action = input(' ')
      if action == 's':
        try:
          bs_points = np.concatenate(self.archive['bs'].values)
          utils.show(bs_points, self.filepath)
        except:
          print('Cannot show progress now.')

  def measure_novelty(self):
    # MEASURE AGENT NOVELTY
    for agent_idx in range(self.pop.size):
      bs_point = self.pop[agent_idx]['bs']

      bs_space = np.concatenate(self.pop['bs'].values)
      bs_space = np.delete(bs_space, agent_idx, axis=0)
      if self.archive.size > 0:
        archive_bs_space = np.concatenate(self.archive['bs'].values)
        bs_space = np.concatenate([bs_space, archive_bs_space])
      # Get distances
      diff = np.atleast_2d(bs_space - bs_point)
      dists = np.sqrt(np.sum(diff * diff, axis=1))
      k = 15
      if len(dists) <= k: # Should never happen
        idx = list(range(len(dists)))
        k = len(idx)
      else:
        idx = np.argpartition(dists, k)  # Get 15 nearest neighs
      novel = True

      mean_k_dist = np.mean(dists[idx[:k]])
      if not self.adaptive_distance:
        self.pop[agent_idx]['surprise'] = mean_k_dist
      else:
        if mean_k_dist <= self.min_dist:
          novel = False
        self.pop[agent_idx]['best'] = novel

  def update_archive(self):
    new_gen = []  # Reproduce only the novel ones

    # ADD AGENT TO ARCHIVE
    if not self.adaptive_distance:
      novel = self.pop['surprise'].sort_values(ascending=False)
      best = novel.iloc[:5].index.values  # Get 5 best
      dead = novel.iloc[-5:].index.values  # Get 5 worst
      if self.archive is not None:
        for idx in best:
          if self.pop[idx]['name'] not in self.archive['name'].values:
            self.archive.add(self.pop.copy(idx, with_data=True))  # Only add the most novel ones
      for i in best:
        new_gen.append(self.pop.copy(i))
        self.pop[i]['best'] = True

    else:
      for agent_idx in range(self.pop.size):
        if self.pop[agent_idx]['best'] and self.pop[agent_idx]['name'] not in self.archive['name'].values:
          if len(self.archive) >= self.max_arch_len: # If archive is full, replace a random element
            replaced = random.randint(0, len(self.archive)-1)
            self.archive[replaced] = self.pop.copy(agent_idx, with_data=True)
          else:
            self.archive.add(self.pop.copy(agent_idx, with_data=True))
          self.novel_in_gen += 1

        elif np.random.uniform() <= 0.005 and self.pop[agent_idx]['name'] not in self.archive['name'].values:
          if len(self.archive) >= self.max_arch_len:
            replaced = random.randint(0, len(self.archive)-1)
            self.archive[replaced] = self.pop.copy(agent_idx, with_data=True)
          else:
            self.archive.add(self.pop.copy(agent_idx, with_data=True))
          # self.novel_in_gen += 1
      for i, a in enumerate(self.pop):
        if a['best']:
          new_gen.append(self.pop.copy(i))
      dead = random.sample(range(self.pop.size), len(new_gen))

    # This one is common in both adaptive and non adaptive distance
    for i, new_agent in zip(dead, new_gen):
      self.pop[i] = new_agent

    # Mutate pop
    for a in self.pop:
      if np.random.random() <= self.mutation_rate:
        a['agent'].mutate()
        a['name'] = self.pop.agent_name  # When an agent is mutated it also changes name, otherwise it will never be added to the archive
        self.pop.agent_name += 1
      a['best'] = False

  def evaluate_agent(self, agent):
    """
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    """
    done = False
    cumulated_reward = 0
    obs = utils.obs_formatting(env_tag, self.env.reset())
    t = 0
    while not done:
      action = utils.action_formatting(env_tag, agent['agent'](t))
      obs, reward, done, info = self.env.step(action)
      obs = utils.obs_formatting(env_tag, obs)
      cumulated_reward += reward
      t += 1
    agent['bs'] = np.array([[obs[0][0], obs[0][1]]])
    agent['reward'] = cumulated_reward

  def evolve(self, gen=1000):
    self.elapsed_gen = 0
    for self.elapsed_gen in range(gen):
      for a in self.pop:
        self.evaluate_agent(a)

      self.measure_novelty()
      self.update_archive()

      if self.novel_in_gen > 2:
        self.min_dist += self.min_dist*0.1
        self.not_added = 0
      elif self.novel_in_gen == 0:
        self.not_added += 1
      else:
        self.not_added = 0
      self.novel_in_gen = 0
      if self.not_added > 4 and self.min_dist > 0.3:
        self.min_dist -= self.min_dist * 0.1

      if self.elapsed_gen % 10 == 0:
        print('Gen {}'.format(self.elapsed_gen))
        print('Archive size {}'.format(self.archive.size))
        print('Min distance {}'.format(self.min_dist))
        print()



if __name__ == '__main__':
  env = gym.make(env_tag)

  env.seed()
  np.random.seed()

  filepath = os.path.join(utils.get_projectpath(), 'baselines', 'ns')

  ns = NoveltySearch(env, filepath, pop_size=100, obs_shape=6, action_shape=2)
  try:
    ns.evolve(500)
  except KeyboardInterrupt:
    print('User Interruption')

  bs_points = np.concatenate(ns.archive['bs'].values)
  utils.show(bs_points, ns.filepath, 'NS_{}_{}'.format(ns.elapsed_gen, env_tag))
  print(ns.archive['name'].values)

  print('Testing result according to best reward.')
  rewards = ns.archive['reward'].sort_values(ascending=False)
  for idx in range(ns.archive.size):
    tested = ns.archive[rewards.iloc[idx:idx + 1].index.values[0]]
    print()
    print('Testing agent {} with reward {}'.format(tested['name'], tested['reward']))
    done = False
    ts = 0
    obs = utils.obs_formatting(env_tag, ns.env.reset())
    while not done and ts < 1000:
      ns.env.render()
      action = utils.action_formatting(env_tag, tested['agent'](ts))
      obs, reward, done, info = ns.env.step(action)
      obs = utils.obs_formatting(env_tag, obs)
      ts += 1
