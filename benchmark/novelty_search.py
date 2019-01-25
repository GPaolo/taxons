import numpy as np
from core.rnd import rnd
from core.qd import population, agents
import gym
import random
import gym_billiard
env_tag = 'MountainCarContinuous-v0'

class NoveltySearch(object):
  def __init__(self, env, obs_shape=6, action_shape=2, pop_size=50):
    self.obs_space = obs_shape
    self.action_space = action_shape
    self.pop = population.Population(agent=agents.FFNeuralAgent,
                                     input_shape=obs_shape,
                                     output_shape=action_shape,
                                     pop_size=pop_size)
    self.archive = population.Population(agent=agents.FFNeuralAgent,
                                         input_shape=obs_shape,
                                         output_shape=action_shape,
                                         pop_size=0)
    self.env = env
    self.min_dist = 0.5
    self.novel_in_gen = 0
    self.not_added = 0
    self.mutation_rate = 0.5

  def novelty(self, agent):
    bs_point = agent['bs']
    # If the archive is empty, add the agent directly
    if self.archive.size == 0:
      agent['best'] = True
      self.archive.add(agent)
      return
    else:
      bs_space = np.concatenate(self.archive['bs'].values)
    # Get distances
    diff = np.atleast_2d(bs_space - bs_point)
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    k = 6
    if self.archive.size <= k:
      idx = list(range(self.archive.size))
      k = len(idx)
    else:
      idx = np.argpartition(dists, k) # Get 5 nearest neighs
    novel = True

    for a in dists[idx[:k]]:
      if not a == 0 and a <= self.min_dist:
        novel = False
        break

    if novel:
      self.archive.add(agent)
      agent['best'] = True
      self.novel_in_gen += 1
    elif np.random.uniform() <= 0.005:
      self.archive.add(agent)
      agent['best'] = True
      self.novel_in_gen += 1

  def evaluate_agent(self, agent):
    '''
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    '''
    done = False
    cumulated_reward = 0
    obs = self.env.reset()
    if env_tag is 'Billiard-v0':
      obs = np.concatenate(obs)
    obs = np.array([obs])
    while not done:
      action = np.squeeze(agent['agent'](obs))
      obs, reward, done, info = self.env.step([action])
      if env_tag is 'Billiard-v0':
        obs = np.concatenate(obs)
      obs = np.array([obs])
      cumulated_reward += reward
    agent['bs'] = np.array([[obs[0][0], 0]])
    agent['reward'] = cumulated_reward

  def show_bs(self):
    print('Behaviour space coverage representation.')
    bs_points = np.concatenate(self.archive['bs'].values)
    import matplotlib.pyplot as plt

    pts = ([x[0] for x in bs_points if x is not None], [y[1] for y in bs_points if y is not None])
    # plt.scatter(pts[0], pts[1])
    plt.hist(pts[0])
    plt.show()

  def evolve(self, gen=1000):
    self.elapsed_gen = 0
    for self.elapsed_gen in range(gen):
      for a in self.pop:
        self.evaluate_agent(a)
        self.novelty(a)

      if self.novel_in_gen > 2:
        self.min_dist += self.min_dist*0.1
        self.not_added = 0
      elif self.novel_in_gen == 0:
        self.not_added += 1
      else:
        self.not_added = 0
      self.novel_in_gen = 0
      if self.not_added > 4 and self.min_dist > 0.2:
        self.min_dist -= self.min_dist * 0.1

      new_gen = []
      for i, a in enumerate(self.pop):
        if a['best']:
          new_gen.append(self.pop.copy(i))

      dead = random.sample(range(self.pop.size), len(new_gen))
      for i, new_agent in zip(dead, new_gen):
        self.pop[i] = new_agent

      # Mutate pop that are not novel
      for a in self.pop:
        if np.random.random() <= self.mutation_rate and not a['best']:
          a['agent'].mutate()
        a['best'] = False

      if self.elapsed_gen % 10 == 0:
        print('Gen {}'.format(self.elapsed_gen))
        print('Archive size {}'.format(self.archive.size))
        print('Min distance {}'.format(self.min_dist))
        print()

      if self.elapsed_gen % 1000 == 0:
        self.show_bs()



if __name__ == '__main__':
  env = gym.make(env_tag)

  env.seed()
  np.random.seed()
  ns = NoveltySearch(env, pop_size=50, obs_shape=2, action_shape=1)
  try:
    ns.evolve(3000)
  except KeyboardInterrupt:
    print('User Interruption')

  ns.show_bs()




