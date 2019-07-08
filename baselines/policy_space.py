# Created by Giuseppe Paolo 
# Date: 30/06/2019

import numpy as np
from core.qd import population, agents
from core.utils import utils
import os, gc, json

class PolicySpace(object):

  # ---------------------------------------------------
  def __init__(self, env, parameters):
    self.params = parameters
    self.pop_size = self.params.pop_size
    self.env = env
    self.save_path = self.params.save_path
    self.agents_shapes = self.params.agent_shapes
    self.agent_name = self.params.qd_agent

    self.logs = {'Generation': [], 'Avg gen surprise': [], 'Max reward': [], 'Archive size': [], 'Coverage': []}

    if self.agent_name == 'Neural':
      agent_type = agents.FFNeuralAgent
    elif self.agent_name == 'DMP':
      agent_type = agents.DMPAgent

    self.population = population.Population(agent=agent_type,
                                            shapes=self.agents_shapes,
                                            pop_size=self.pop_size)
    self.archive = None
    if self.params.use_archive:
      self.archive = population.Population(agent=agent_type,
                                           shapes=self.agents_shapes,
                                           pop_size=0)

    self.opt = self.params.optimizer(self.population, archive=self.archive, mutation_rate=self.params.mutation_rate,
                                     metric_update_interval=self.params.update_interval)

    self.END = False
    self.elapsed_gen = 0
    # ---------------------------------------------------

  # ---------------------------------------------------
  def evaluate_agent(self, agent):
    """
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    """
    done = False
    cumulated_reward = 0

    obs = utils.obs_formatting(self.params.env_tag, self.env.reset())
    t = 0
    while not done:
      agent_input = t
      action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input))
      obs, reward, done, info = self.env.step(action)
      obs = utils.obs_formatting(self.params.env_tag, obs)
      t += 1
      cumulated_reward += reward

      if 'Ant' in self.params.env_tag:
        CoM = np.array([self.env.env.data.qpos[:2]])
        if t >= self.params.max_episode_len or np.any(np.abs(CoM) >= np.array([4, 4])):
          done = True

    if 'Ant' in self.params.env_tag:
      agent['bs'] = np.array([self.env.env.data.qpos[:2]])  # xy position of CoM of the robot
    else:
      agent['bs'] = np.array([[obs[0][0], obs[0][1]]])
    agent['reward'] = cumulated_reward

    # Extract genome as a feature
    feat = []
    for k in agent['agent'].genome:
      if isinstance(k, dict):
        for i in k:
          if i is not 'name':
            feat.append(k[i])
      else:
        feat.append(k)

    agent['features'] = [np.array(feat), None] #TODO check this!!!
    return cumulated_reward
  # ---------------------------------------------------

  # ---------------------------------------------------
  def train(self, steps=10000):
    for self.elapsed_gen in range(steps):
      for agent in self.population:
        self.evaluate_agent(agent)

      max_rew = np.max(self.population['reward'].values)
      self.opt.step()

      if self.elapsed_gen % 10 == 0:
        gc.collect()
        print('Seed {} - Generation {}'.format(self.params.seed, self.elapsed_gen))
        if self.archive is not None:
          print('Seed {} - Archive size {}'.format(self.params.seed, self.archive.size))
        print('Seed {} - Max reward {}'.format(self.params.seed, max_rew))
        print('Saving checkpoint...')
        self.save(ckpt=True)
        print("Done")
        print()

      if self.archive is not None:
        bs_points = np.concatenate(self.archive['bs'].values)
      else:
        bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
      if 'Ant' in self.params.env_tag:
        limit = 5
      else:
        limit = 1.35
      coverage = utils.show(bs_points, filepath=self.save_path, info={'gen':self.elapsed_gen, 'seed':self.params.seed}, limit=limit)

      self.logs['Generation'].append(str(self.elapsed_gen))
      self.logs['Avg gen surprise'].append('0')
      self.logs['Max reward'].append(str(max_rew))
      self.logs['Archive size'].append(str(self.archive.size))
      self.logs['Coverage'].append(str(coverage))
      if self.END:
        print('Seed {} - Quitting.'.format(self.params.seed))
        break
    gc.collect()
  # ---------------------------------------------------

  # ---------------------------------------------------
  def save(self, ckpt=False):
    if ckpt:
      folder = 'models/ckpt'
    else:
      folder = 'models'
    save_subf = os.path.join(self.save_path, folder)
    print('Seed {} - Saving...'.format(self.params.seed))
    if not os.path.exists(save_subf):
      try:
        os.makedirs(os.path.abspath(save_subf))
      except:
        print('Seed {} - Cannot create save folder.'.format(self.params.seeds))
    self.population.save_pop(save_subf, 'pop')
    self.archive.save_pop(save_subf, 'archive')

    with open(os.path.join(self.save_path, 'logs.json'), 'w') as f:
      json.dump(self.logs, f, indent=4)
    print('Seed {} - Done'.format(self.params.seed))
  # ---------------------------------------------------