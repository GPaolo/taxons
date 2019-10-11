# Created by giuseppe
# Date: 11/10/19

import numpy as np

# ---------------------------------------------------------------------------
class FCLayer(object):
  """
  This one is a simple FC layer to be used as genome of the evolution agents
  """
  # ------------------------------------------------------
  def __init__(self, input, output, name='fc', bias=True):
    """
    Constructor
    :param input: input shape
    :param output: output shape
    :param name: name of the layer
    :param bias: flag to select if to use or not the bias
    """
    self.w = np.random.randn(input, output)
    if bias:
      self.bias = np.random.randn(1, output)
    else:
      self.bias = np.zeros((1, output))
    self.name = name
  # ------------------------------------------------------

  # ------------------------------------------------------
  def __call__(self, x):
    """
    Call function
    :param x: input
    :return: The output of the layer
    """
    return self.forward(x)
  # ------------------------------------------------------

  # ------------------------------------------------------
  def forward(self, x):
    """
    Forward function
    :param x: input
    :return: The output of the layer
    """
    assert np.shape(x)[0] == 1, 'Wrong input shape. Needs to be [1, n]. Is {}'.format(np.shape(x))
    return np.matmul(x, self.w) + self.bias
  # ------------------------------------------------------

  # ------------------------------------------------------
  @property
  def params(self):
    """
    Parameters of the layer.
    :return: A dict with weights and bias
    """
    return {'w': self.w, 'bias': self.bias}
  # ------------------------------------------------------

  # ------------------------------------------------------
  @property
  def show(self):
    """
    Prints the parameters of the layer
    """
    print('Layer {}:'.format(self.name))
    print('Weights {}'.format(self.w))
    print('Bias {}'.format(self.bias))
  # ------------------------------------------------------

  # ------------------------------------------------------
  def load(self, params):
    """
    Load function
    :param params: Parameters of the layer
    """
    self.w = params['w']
    self.bias = params['bias']
  # ------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class DMPExp(object):
  """
  This one is a exponential DMP
  """
  # ------------------------------------------------------
  def __init__(self, name='dmp', degree=5):
    """
    Constructor
    :param name: name
    :param degree: number of basis functions to use
    """
    self.num_basis_func = degree
    self.mu = np.abs(np.random.randn(self.num_basis_func))
    self.sigma = np.random.uniform(size=self.num_basis_func)
    self.w = np.random.randn(self.num_basis_func)
    self.a_x  = np.random.uniform()
    self.tau = 2
    self.name = name
  # ------------------------------------------------------

  # ------------------------------------------------------
  def __call__(self, t):
    """
    Call function
    :param t: Input, the time
    :return: The value of the DMP
    """
    g = np.zeros(self.num_basis_func)
    x = np.exp( -self.a_x*t/self.tau )

    for k in range(self.num_basis_func):
      g[k] = self.basis_function(x, self.mu[k], self.sigma[k])
    f = np.sum(g*self.w)/np.sum(g)
    return f
  # ------------------------------------------------------

  # ------------------------------------------------------
  @property
  def params(self):
    """
    Parameters of the DMP.
    :return: A dict of the parameters
    """
    return {'mu': self.mu, 'sigma': self.sigma, 'w': self.w, 'a_x': self.a_x, 'tau': self.tau, 'name':self.name}
  # ------------------------------------------------------

  # ------------------------------------------------------
  @staticmethod
  def basis_function(x, mu, sigma):
    """
    This basis function samples the value of gaussian in x.
    :param x: Point where to sample the gaussian
    :param mu: Center of the gaussian
    :param sigma: Std deviation
    :return: The value of the basis function
    """
    return np.exp(np.sin((x - mu)/sigma)/2)
  # ------------------------------------------------------

  # ------------------------------------------------------
  def load(self, params):
    """
    Load function
    :param params: Parameters of the DMP
    """
    self.mu = params['mu']
    self.sigma = params['sigma']
    self.w = params['w']
    self.a_x = params['a_x']
    self.tau = params['tau']
    self.name = params['name']
  # ------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class DMPPoly(object):
  """
  This one is a Polynomial DMP
  """
  # ------------------------------------------------------
  def __init__(self, name='dmp', degree=5):
    """
    Constructor
    :param name: name of the dmp
    :param kwargs: number of basis functions
    """
    self.degree = degree
    self.w = np.random.randn(self.degree+1)
    self.scale = 1
    self.name = name
  # ------------------------------------------------------

  # ------------------------------------------------------
  def __call__(self, t):
    """
    Call function
    :param t: Input, the time
    :return: The value of the DMP
    """
    x = np.cos(t/self.scale) # This way we limit it between [-1, 1]
    p = 0
    for i in range(self.degree+1):
      p += self.w[i]*x**i
    return p
  # ------------------------------------------------------

  # ------------------------------------------------------
  @property
  def params(self):
    """
    Parameters of the DMP.
    :return: A dict of the parameters
    """
    return {'w': self.w, 'scale':self.scale, 'name':self.name}
  # ------------------------------------------------------

  # ------------------------------------------------------
  def load(self, params):
    """
    Load function
    :param params: Parameters of the DMP
    """
    self.scale = params['scale']
    self.w = params['w']
    self.name = params['name']
  # ------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class DMPSin(object):
  """
  This one is a Sinusoidal DMP
  """

  # ------------------------------------------------------
  def __init__(self, name='dmp', *kwargs):
    """
    Constructor
    :param name: Name of the dmp
    :param kwargs:
    """
    self.name = name
    self.amplitude = np.random.randn()
    self.period = np.random.uniform(10, 50) # self.period = np.random.uniform() # MUJOCO
  # ------------------------------------------------------

  # ------------------------------------------------------
  def __call__(self, t):
    """
    Call function
    :param t: Input, the time.
    :return: The value of the sin
    """
    x = self.amplitude * np.sin(2*np.pi*t/self.period)
    return x
  # ------------------------------------------------------

  # ------------------------------------------------------
  @property
  def params(self):
    """
    Parameters of the DMP.
    :return: A dict of the parameters
    """
    return {'period': self.period, 'amplitude': self.amplitude, 'name':self.name}
  # ------------------------------------------------------

  # ------------------------------------------------------
  @property
  def amplitude(self):
    """
    Amplitude of the sinusoid. Is defined as a property cause there are only a bunch of values that it can have.
    """
    return self.__amplitude

  @amplitude.setter
  def amplitude(self, x):
    """
    Amplitude setter
    :param x: Value
    """
    if x > 5:
      self.__amplitude = 5
    elif x < -5:
      self.__amplitude = -5
    else:
      self.__amplitude = x
  # ------------------------------------------------------

  # ------------------------------------------------------
  def load(self, params):
    """
    Load function
    :param params: Parameters of the DMP
    """
    self.period = params['period']
    self.amplitude = params['amplitude']
    self.name = params['name']
  # ------------------------------------------------------
# ---------------------------------------------------------------------------
