import numpy as np
from core.qd.agents import *


class Population(object):
  '''
  Basic Population class. The new generation is just the mutation of the best elements that substitutes the worst.
  The criteria for the best is given by the metric, and is calculated outside.
  '''
  def __init__(self, agent=BaseAgent, pop_size=10, max_len=None, *args, **kargs):
    self.size = pop_size
    self.pop = []
    self.agent_class = agent
    self.kargs = kargs
    self.max_len = max_len

    for i in range(self.size):
      self.add()

  # These 3 functions allow to work with the pop as a list
  def __iter__(self):
    '''
    Allows to directly iterate the pop.
    :return:
    '''
    return self.pop.__iter__()

  def __getitem__(self, item):
    return self.pop.__getitem__(item)

  def __setitem__(self, key, value):
    assert set(self.pop[0]) == set(value), 'Wrong agent dict keys.'
    return self.pop.__setitem__(key, value)

  def _append(self, item):
    assert set(self.pop[0]) == set(item), 'Wrong agent dict keys.'
    self.pop.append(item)

  def add(self, agent=None):
    '''
    Adds agent to the pop. If no agent is passed, a new agent is generated.
    :param agent: agent to add
    :return:
    '''
    if agent is not None:
      self._append(agent)
    else:
      agent = {'agent': self.agent_class(self.kargs), 'reward': None, 'surprise': None, 'best': False, 'bs':None}
      self.pop.append(agent)

    if self.max_len is not None:
      while len(self.pop) > self.max_len:
        del self.pop[self.max_len]

  def show(self):
    print(self.pop)




if __name__ == '__main__':
  pop = Population(agent=FFNeuralAgent, input_shape=3, output_shape=3, pop_size=3)

  len(pop)

  a = pop[0:1]
  print(a)



