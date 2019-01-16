import numpy as np
from core.qd.agents import *
import pandas as pd

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

  def copy(self, idx):
    assert idx < self.size and idx > -self.size-1, 'Index out of range'
    agent = {'agent': self.agent_class(self.kargs), 'reward': None, 'surprise': None, 'best': False, 'bs': None}
    agent['agent'] = deepcopy(self[idx]['agent'])
    agent = pd.DataFrame([agent], columns=agent.keys())
    return agent.iloc[0]








if __name__ == '__main__':
  pop = Population(agent=FFNeuralAgent, input_shape=3, output_shape=3, pop_size=3)

  a = pop.copy(-1)
  a['best']=True
  pop.show()
  pop.add(a)
  pop.show()



