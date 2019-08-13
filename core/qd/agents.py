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
      self.sigma = 0.05
      self.mu = 0.
      def normal(*args):
        return self.sigma * np.random.randn(*args) + self.mu

      self.mutation_operator = normal
    else:
      self.mutation_operator = mutation_distr
    self.action_len = 0.
    self._genome = []

  def evaluate(self, x):
    raise NotImplementedError

  @property
  def genome(self):
    gen = [l.params for l in self._genome]
    gen.append(self.action_len)
    return gen

  @property
  def action_len(self):
    return self._action_len

  @action_len.setter
  def action_len(self, l):
    if l < 0.:
      self._action_len = 0.
    elif l > 1.:
      self._action_len = 1.
    else:
      self._action_len = l

  def mutate(self):
    raise NotImplementedError

  def copy(self):
    '''
    Does a deep copy of the agent
    :return:
    '''
    return deepcopy(self)

  def load_genome(self, genome, agent_name):
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
    self.use_bias = True

    self.action_len = np.random.uniform(0.5, 1)
    self._genome = [utils.FCLayer(self.input_shape, 5, 'fc1', bias=self.use_bias),
                    utils.FCLayer(5, self.output_shape, 'fc2', bias=self.use_bias)]

  def evaluate(self, x):
    # TODO REMOVE THIS OBSCENITY.
    output = x[0]
    t = x[1]
    if not len(np.shape(output)) > 1:
      output = np.array([output])

    if t > self.action_len:
     output = [np.zeros(self.output_shape)]
    else:
      for l in self._genome[:-1]:
        output = self.sigmoid(l(output))
      output = np.tanh(self._genome[-1](output))
    return output

  def sigmoid(self, x):
    return np.exp(-np.logaddexp(0, -x))

  def __call__(self, x):
    return self.evaluate(x)

  def mutate(self):
    '''
    Mutates the genome of the agent. It does not return anything. The mutation is internal.
    :return:
    '''
    for l in self._genome:
      self._mutate_layer(l)
    self.action_len = np.clip(self.action_len + self.mutation_operator(), a_min=0.5, a_max=1)

  def _mutate_layer(self, layer):
    mutation_selection = np.array(np.random.uniform(size=(layer.w.shape[0], layer.w.shape[1]))<= 0.2).astype(int)
    layer.w += self.mutation_operator(layer.w.shape[0], layer.w.shape[1]) * mutation_selection
    layer.w = np.clip(layer.w, a_min=-5, a_max=5)

    if self.use_bias:
      mutation_selection = np.array(np.random.uniform(size=(layer.bias.shape[0], layer.bias.shape[1]))<= 0.2).astype(int)
      layer.bias += self.mutation_operator(layer.bias.shape[0], layer.bias.shape[1]) * mutation_selection


  def load_genome(self, params, agent_name):
    self.action_len = params[-1] # the last is the action lenght

    for p, g in zip(params[:-1], self._genome):
      assert np.all(np.shape(g.w) == np.shape(p['w'])), 'Wrong shape of weight for layer {} of agent {}'.format(self.name, agent_name)
      assert np.all(np.shape(g.bias) == np.shape(p['bias'])), 'Wrong shape of bias for layer {} of agent {}'.format(self.name, agent_name)
      g.w = p['w']
      g.bias = p['bias']


class DMPAgent(BaseAgent):

  def __init__(self, shapes, mutation_distr=None):
    super(DMPAgent, self).__init__(mutation_distr)

    self.dof = shapes['dof']
    self.shapes = shapes
    self.action_len = np.random.uniform(0.5, 1)

    self._genome = []
    if shapes['type'] == 'poly':
      _dmp = utils.DMPPoly
    elif shapes['type'] == 'exp':
      _dmp = utils.DMPExp
    elif shapes['type'] == 'sin':
      _dmp = utils.DMPSin
    else:
      print('Wrong DMP chosen')
      raise ValueError

    for i in range(self.dof):
      self._genome.append(_dmp('dmp{}'.format(i), **shapes))

  def evaluate(self, x):
    output = np.zeros(self.dof)

    if x <= self.action_len:
      for i, dmp in enumerate(self._genome):
        output[i] = dmp(x)

    return [output]

  def __call__(self, x):
    return self.evaluate(x)

  def mutate(self):
    for dmp in self._genome:
      for param_name in dmp.params:
        if param_name == 'name':
          continue
        try:
          new_value = dmp.params[param_name] + self.mutation_operator(dmp.params[param_name].shape[0])
        except AttributeError:
          new_value = dmp.params[param_name] + self.mutation_operator()
        setattr(dmp, param_name, new_value)
    self.action_len = self.action_len + self.mutation_operator()

  def load_genome(self, params, agent):
    self.action_len = params[-1]  # the last is the action lenght

    for p, g in zip(params[:-1], self._genome):
      g.load(deepcopy(p))




if __name__ == '__main__':
  agent = DMPAgent({'degree':5, 'dof':1})
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
  for k in range(ts):
    f = agent(k)
    a.append(f[0])
  agent.mutate()
  for k in range(ts):
    f = agent(k)
    b.append(f[0])

  print(len(a))
  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  ax1.plot(list(range(ts)), b)
  ax1.plot(list(range(ts)), a)
  plt.show()

