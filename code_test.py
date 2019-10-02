# Created by Giuseppe Paolo 
# Date: 26/03/19

import numpy as np
import deap
from deap import base, creator, tools

class BaseAgent(object):

  def __init__(self, mutation_distr=None, **kwargs):
    '''
    This class defines the base agent from which other agents should inherit
    '''
    if mutation_distr is None:
      # Define normal distr with sigma and mu
      self.sigma = 0.05
      self.mu = 0.
      def normal(*args):
        return self.sigma * np.random.randn(*args) + self.mu

      self.mutation_operator = normal
    else:
      self.mutation_operator = mutation_distr
    self.action_len = 0.
    self._genome = []

  def print(self):
    print(0)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create('Individual', BaseAgent, mutation_distribution=None, fitness=creator.FitnessMin)

POP_SIZE = 3
toolbox = base.Toolbox()
toolbox.register("individual", creator.Individual)
toolbox.register("select", tools.selTournament, tournsize=3)


# toolbox.register("population", tools.initRepeat, list, creator.Individual)
#
# pop = toolbox.population(n=POP_SIZE)
# archive = toolbox.population(n=0)
ind = toolbox.individual()
print(ind)
print(ind.fitness.valid)


