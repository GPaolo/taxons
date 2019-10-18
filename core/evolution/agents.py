import numpy as np
from core.evolution import genome
from copy import deepcopy

# ---------------------------------------------------------------------------
class BaseAgent(object):
  """
  This class defines the base agent from which other agents should inherit
  """
  # ---------------------------------
  def __init__(self, mutation_distr=None, **kwargs):
    """
    Constructor
    :param mutation_distr: Distribution used to mutate the agent genome. If None, a normal distribution, with [0, 0.05] is used
    :param kwargs:
    """
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
  # ---------------------------------

  # ---------------------------------
  def evaluate(self, *args):
    """
    Evaluate agent. Needs to be implemented by inheriting classes
    :param x: Input
    """
    raise NotImplementedError
  # ---------------------------------

  # ---------------------------------
  def __call__(self, *args):
    """
    Call function
    :param x: Agent input
    :return: Value of the output
    """
    return self.evaluate(*args[0])
  # ---------------------------------

  # ---------------------------------
  @property
  def genome(self):
    """
    Genome of the agent in the shape of a list
    :return: The genome
    """
    gen = [l.params for l in self._genome]
    gen.append(self.action_len)
    return gen
  # ---------------------------------

  # ---------------------------------
  @property
  def action_len(self):
    """
    Length of the agent actions.
    """
    return self._action_len

  @action_len.setter
  def action_len(self, l):
    if l < 0.:
      self._action_len = 0.
    elif l > 1.:
      self._action_len = 1.
    else:
      self._action_len = l
  # ---------------------------------

  # ---------------------------------
  def mutate(self):
    """
    Mutate agent genome. Needs to be implemented by inheriting classes
    """
    raise NotImplementedError
  # ---------------------------------

  # ---------------------------------
  def copy(self):
    """
    Performs a deep copy of the agent
    :return: A copy of the agent
    """
    return deepcopy(self)
  # ---------------------------------

  # ---------------------------------
  def load_genome(self, genome, agent_name):
    """
    Loads a genome into the agent. Needs to be implemented by inheriting classes
    :param genome: Genome to load
    :param agent_name: Name of the agent
    """
    raise NotImplementedError
  # ---------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
class FFNeuralAgent(BaseAgent):
  """
  This agent implements a small feedforward neural network agent.
  """
  # ---------------------------------
  def __init__(self, shapes, mutation_distr=None):
    """
    Constructor
    :param mutation_distr: distribution used for mutation
    :param shapes: Dict containing
              input_shape: shape of network input
              output_shape: shape of network output
    """
    super(FFNeuralAgent, self).__init__(mutation_distr)

    self.input_shape = shapes['input_shape']
    self.output_shape = shapes['output_shape']
    self.use_bias = True

    self.action_len = np.random.uniform(0.5, 1)
    self._genome = [genome.FCLayer(self.input_shape, 5, 'fc1', bias=self.use_bias),
                    genome.FCLayer(5, self.output_shape, 'fc2', bias=self.use_bias)]
  # ---------------------------------

  # ---------------------------------
  def evaluate(self, t, obs):
    """
    Evaluate agent.
    :param t: Time step
    :param obs: Observation
    """
    output = obs
    if not len(np.shape(output)) > 1:
      output = np.array([output])

    if t > self.action_len:
     output = [np.zeros(self.output_shape)]
    else:
      for l in self._genome[:-1]:
        output = self.sigmoid(l(output))
      output = np.tanh(self._genome[-1](output))
    return output
  # ---------------------------------

  # ---------------------------------
  def sigmoid(self, x):
    """
    Sigmoid function
    :param x: Input
    """
    return np.exp(-np.logaddexp(0, -x))
  # ---------------------------------

  # ---------------------------------
  def mutate(self):
    """
    Mutates the genome of the agent. It does not return anything. The mutation is internal.
    """
    for l in self._genome:
      self._mutate_layer(l)
    self.action_len = np.clip(self.action_len + self.mutation_operator(), a_min=0.5, a_max=1)
  # ---------------------------------

  # ---------------------------------
  def _mutate_layer(self, layer):
    """
    Mutates single layer of the network
    :param layer: Layer to mutate
    """
    mutation_selection = np.array(np.random.uniform(size=(layer.w.shape[0], layer.w.shape[1]))<= 0.2).astype(int)
    layer.w += self.mutation_operator(layer.w.shape[0], layer.w.shape[1]) * mutation_selection
    layer.w = np.clip(layer.w, a_min=-5, a_max=5)

    if self.use_bias:
      mutation_selection = np.array(np.random.uniform(size=(layer.bias.shape[0], layer.bias.shape[1]))<= 0.2).astype(int)
      layer.bias += self.mutation_operator(layer.bias.shape[0], layer.bias.shape[1]) * mutation_selection
  # ---------------------------------

  # ---------------------------------
  def load_genome(self, params, agent_name):
    """
    This loads the genome of the agent
    :param params: Genome to load
    :param agent_name: Name of the agent
    """
    self.action_len = params[-1] # the last is the action lenght

    for p, g in zip(params[:-1], self._genome):
      assert np.all(np.shape(g.w) == np.shape(p['w'])), 'Wrong shape of weight for layer {} of agent {}'.format(self.name, agent_name)
      assert np.all(np.shape(g.bias) == np.shape(p['bias'])), 'Wrong shape of bias for layer {} of agent {}'.format(self.name, agent_name)
      g.load(deepcopy(p))
  # ---------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class DMPAgent(BaseAgent):
  """
  This class implements DMP agents
  """
  # ---------------------------------
  def __init__(self, shapes, mutation_distr=None):
    """
    :param mutation_distr: distribution used for mutation
    :param shapes: Dict containing
              dof: output size
              degree: degree of the DMP
              type: Typology of the DMP: poly, exp, sin
    """
    super(DMPAgent, self).__init__(mutation_distr)

    self.dof = shapes['dof']
    self.shapes = shapes
    self.action_len = np.random.uniform(0.5, 1)

    self._genome = []
    if shapes['type'] == 'poly':
      _dmp = genome.DMPPoly
    elif shapes['type'] == 'exp':
      _dmp = genome.DMPExp
    elif shapes['type'] == 'sin':
      _dmp = genome.DMPSin
    else:
      print('Wrong DMP chosen')
      raise ValueError

    for i in range(self.dof):
      self._genome.append(_dmp('dmp{}'.format(i), shapes['degree']))
  # ---------------------------------

  # ---------------------------------
  def evaluate(self, t):
    """
    Evaluate agent.
    :param t: Time step
    :return: Returns output of the Agent
    """
    output = np.zeros(self.dof)

    if t/500. <= self.action_len: #if x <= self.action_len: MUJOCO
      for i, dmp in enumerate(self._genome):
        output[i] = dmp(t)

    return [output]
  # ---------------------------------

  # ---------------------------------
  def mutate(self):
    """
    Mutates the genome of the agent. It does not return anything. The mutation is internal.
    """
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
  # ---------------------------------

  # ---------------------------------
  def load_genome(self, params, agent_name):
    """
    Loads the genome of the agent
    :param params: Genome to load
    :param agent_name: Name of the agent
    :return:
    """
    self.action_len = params[-1]  # the last is the action lenght

    for p, g in zip(params[:-1], self._genome):
      g.load(deepcopy (p))
  # ---------------------------------




if __name__ == '__main__':
  agent = DMPAgent({'degree':5, 'dof':1, 'type':'poly'})
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

