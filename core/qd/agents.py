import numpy as np
from core.utils import utils
from abc import ABCMeta, abstractmethod # This is to force implementation of child class methods
from copy import deepcopy

class BaseAgent(metaclass=ABCMeta):

  def __init__(self, mutation_distr=None, **kwargs):
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

  def load_genome(self, genome):
    raise NotImplementedError

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

  def load_genome(self, params, agent_name):
    for p, g in zip(params, self.genome):
      assert np.all(np.shape(g.w) == np.shape(p['w'])), 'Wrong shape of weight for layer {} of agent {}'.format(self.name, agent_name)
      assert np.all(np.shape(g.bias) == np.shape(p['bias'])), 'Wrong shape of bias for layer {} of agent {}'.format(self.name, agent_name)
      g.w = p['w']
      g.bias = p['bias']


class DMPAgent(BaseAgent):

  def __init__(self, shapes, mutation_distr=None):
    super(DMPAgent, self).__init__(mutation_distr)

    self.genome = []
    self.dof = shapes['dof']
    self.num_basis_func = shapes['num_basis_func']

    for i in range(self.dof):
      self.genome.append(utils.DMP(self.num_basis_func, 'dmp{}'.format(i)))

  def evaluate(self, x):
    output = np.zeros(self.dof)
    for i, dmp in enumerate(self.genome):
      output[i] = dmp(x)
    return [output]

  def __call__(self, x):
    return self.evaluate(x)

  def mutate(self):
    for dmp in self.genome:
      dmp.w = dmp.w + self.mutation_operator(dmp.w.shape[0])
      dmp.mu = dmp.mu + self.mutation_operator(dmp.w.shape[0])
      dmp.sigma = dmp.sigma + self.mutation_operator(dmp.w.shape[0])
      dmp.a_x = dmp.a_x + self.mutation_operator()
      dmp.tau = dmp.tau + self.mutation_operator()

  def load_genome(self, params, agent):
    for p, g in zip(params, self.genome):
      assert np.all(np.shape(g.w) == np.shape(p['w'])), 'Wrong shape of weight for dmp {} of agent {}'.format(self.name, agent)
      assert np.all(np.shape(g.sigma) == np.shape(p['sigma'])), 'Wrong shape of sigma for dmp {} of agent {}'.format(self.name, agent)
      assert np.all(np.shape(g.mu) == np.shape(p['mu'])), 'Wrong shape of mu for dmp {} of agent {}'.format(self.name, agent)
      g.w = p['w']
      g.sigma = p['sigma']
      g.mu = p['mu']
      g.tau = p['tau']
      g.a_x = p['a_x']




if __name__ == '__main__':
  agent = DMPAgent({'dof':1, 'num_basis_func':20})
  import gym, gym_billiard

  env = gym.make('Billiard-v0')
  env.seed()

  # t = 0
  # done=False
  # obs = utils.obs_formatting('Billiard-v0', env.reset())
  # while not done:
  #   action = utils.action_formatting('Billiard-v0', agent(t))
  #   t += 1
  #   print(action)
  #   obs, reward, done, info = env.step(action)
  #   obs = utils.obs_formatting('Billiard-v0', obs)
  #   env.render()


  a = []
  b = []
  ts = 1000
  # for k in range(ts):
    # f = agent.genome[0].basis_function(k, 0, 0.1)
    # a.append(f)
  agent.mutate()
  for k in range(ts):
    f = agent(k)
    b.append(f[0])

  print(len(a))
  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  ax1.plot(list(range(ts)), b)
  # ax1.plot(list(range(ts)), b)
  plt.show()

