import numpy as np
from core.qd.agents import *
import pandas as pd
import os
import pickle as pkl

class Population(object):
  '''
  Basic Population class. The new generation is just the mutation of the best elements that substitutes the worst.
  The criteria for the best is given by the metric, and is calculated outside.
  '''
  def __init__(self, agent=BaseAgent, pop_size=10, max_len=None, *args, **kargs):
    self.pop = pd.DataFrame(columns=['agent', 'reward', 'surprise', 'best', 'bs'])
    self.agent_class = agent
    self.kargs = kargs
    self.max_len = max_len
    self.avg_surprise = 0

    for i in range(pop_size):
      self.add()

  # These functions allow to work with the pop as a list
  def __iter__(self):
    '''
    Allows to directly iterate the pop.
    :return:
    '''
    self._iter_idx = 0
    return self

  def __next__(self):
    if self._iter_idx < self.size:
      x = self.pop.loc[self._iter_idx]
      self._iter_idx += 1
    else:
      raise StopIteration
    return x

  def __getitem__(self, item):
    if type(item) is str:
      return self.pop[item]
    return self.pop.iloc[item]

  def __setitem__(self, key, value):
    assert key < self.size and key > -self.size-1, 'Index out of range'
    self.pop.iloc[key] = value

  def __len__(self):
    return self.size

  @property
  def size(self):
    return len(self.pop)

  def add(self, agent=None):
    '''
    Adds agent to the pop. If no agent is passed, a new agent is generated.
    :param agent: agent to add
    :return:
    '''
    if agent is None:
      agent = {'agent': self.agent_class(self.kargs), 'reward': None, 'surprise': None, 'best': False, 'bs':None}

    agent = pd.DataFrame([agent], columns=agent.keys())
    self.pop = pd.concat([self.pop, agent], ignore_index=True, sort=True)

  def show(self):
    for a in self:
      print(a)

  def copy(self, idx, with_data=False):
    assert idx < self.size and idx > -self.size-1, 'Index out of range'
    agent = {'agent': self.agent_class(self.kargs), 'reward': None, 'surprise': None, 'best': False, 'bs': None}
    if with_data:
      for key in agent.keys():
        agent[key] = deepcopy(self[idx][key])
    else:
      agent['agent'] = deepcopy(self[idx]['agent'])

    agent = pd.DataFrame([agent], columns=agent.keys())
    return agent.iloc[0]

  def save_pop(self, filepath, name):
    save_ckpt = {}
    for i, a in enumerate(self['agent']):
      genome = a.get_genome()
      save_ckpt[str(i)] = [(l.params) for l in genome]
    try:
      pkl.dump(save_ckpt, os.path.join(filepath, 'qd_{}.pkl'.format(name)))
    except:
      print('Cannot Save {}.'.format(name))






if __name__ == '__main__':
  pop = Population(agent=FFNeuralAgent, input_shape=3, output_shape=3, pop_size=3)

  kk = pop.save_pop('a')
  print()



