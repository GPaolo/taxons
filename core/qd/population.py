import numpy as np
from core.qd.agents import *
import copy


class Population(object):
  '''
  Basic Population class. The new generation is just the mutation of the best elements that substitutes the worst.
  The criteria for the best is given by the metric, and is calculated outside.
  '''
  def __init__(self, agent=BaseAgent, pop_size=10, mutation_rate=0.9, *args, **kargs):
    self.size = pop_size
    self.pop = []
    self.agent_class = agent
    self.mutation_rate = mutation_rate

    for i in range(self.size):
      agent = {'agent':self.agent_class(kargs), 'reward':None, 'best':False, 'surprise':None}
      self.pop.append(agent)

  # def new_generation(self):
  #   # Get best elements copies and mutate them
  #   best_idx = [i for i, x in enumerate(self.pop['best']) if x]
  #   mutants = [copy.deepcopy(self.pop['agents'][b]) for b in best_idx]
  #   for m in mutants:
  #     m.mutate()







if __name__ == '__main__':
  pop = Population(agent=NeuralAgent, input_shape=3, output_shape=3)

  for i in [1,4,5]:
    pop.pop['best'][i] = True
  # print(pop.pop)
  m = pop.new_generation()
  x = np.ones((1, 3))
  print(m[0](x))
  for p in m:
    p.mutate()
  print(m[0](x))



