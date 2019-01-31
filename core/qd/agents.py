import numpy as np
from core import utils
from abc import ABCMeta, abstractmethod # This is to force implementation of child class methods
from copy import deepcopy

class BaseAgent(metaclass=ABCMeta):

  def __init__(self, mutation_distr=None):
    '''
    This class defines the base agent from which other agents should inherit
    '''
    if mutation_distr is None:
      # Define normal distr with sigma and mu
      self.sigma = 0.2
      self.mu = 0
      def normal(*args):
        return self.sigma * np.random.randn(*args) + self.mu

      self.mutation_operator = normal
    else:
      self.mutation_operator = mutation_distr

    self.genome = None

  def evaluate(self, x):
    raise NotImplementedError

  def get_genome(self):
    return self.genome

  def mutate(self):
    raise NotImplementedError

  def copy(self):
    '''
    Does a deep copy of the agent
    :return:
    '''
    return deepcopy(self)


class FFNeuralAgent(BaseAgent):

  def __init__(self, shapes, mutation_distr=None):
    '''
    This agent embeds an NN. Not using pytorch cause it does not give any advantage (cannot parallelize on one GPU)
    :param mutation_distr: distribution used for mutation
    :param shapes: Dict that has to contain
              input_shape: shape of network input
              output_shape: shape of network output
    '''
    super(FFNeuralAgent, self).__init__(mutation_distr)

    self.input_shape = shapes['input_shape']
    self.output_shape = shapes['output_shape']

    self.fc1 = utils.FCLayer(self.input_shape, 16, 'fc1')
    self.fc2 = utils.FCLayer(16, 32, 'fc2')
    self.fc3 = utils.FCLayer(32, 32, 'fc3')
    self.fc4 = utils.FCLayer(32, 32, 'fc4')
    self.fc5 = utils.FCLayer(32, 16, 'fc5')
    self.fc6 = utils.FCLayer(16, self.output_shape, 'fc6')

    self.genome = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]

  def evaluate(self, x):
    x = np.tanh(self.fc1(x))
    x = np.tanh(self.fc2(x))
    x = np.tanh(self.fc3(x))
    x = np.tanh(self.fc4(x))
    x = np.tanh(self.fc5(x))
    x = np.tanh(self.fc6(x))
    return x

  def __call__(self, x):
    return self.evaluate(x)

  def mutate(self):
    '''
    Mutates the genome of the agent. It does not return anything. The mutation is internal.
    :return:
    '''
    for l in self.genome:
      l.w = l.w + self.mutation_operator(l.w.shape[0], l.w.shape[1])
      l.bias = l.bias + self.mutation_operator(l.bias.shape[0], l.bias.shape[1])



if __name__ == '__main__':
  net = FFNeuralAgent({'input_shape':2, 'output_shape':3})
  net.fc1.show

