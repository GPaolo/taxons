import numpy as np


class FCLayer(object):
  def __init__(self, input, output, name='fc'):
    super().__init__()
    std = np.random.uniform()
    self.w = np.random.randn(input, output) * std
    self.bias = np.random.randn(1, output)
    self.name = name

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    assert np.shape(x)[0] == 1, 'Wrong input shape. Needs to be [1, n]. Is {}'.format(np.shape(x))
    return np.matmul(x, self.w) + self.bias

  @property
  def params(self):
    return self.w, self.bias

  @property
  def show(self):
    print('Layer {}:'.format(self.name))
    print('Weights {}'.format(self.w))
    print('Bias {}'.format(self.bias))


def action_formatting(env_tag, action):
  """
  This function helps reformat the actions according to the environment
  :param env_tag: Environment name
  :param action: Action to reformat
  :return:
  """
  if env_tag == 'MountainCarContinuous-v0':
    assert action.shape == (1,1), 'Shape is not of dimension {}. Has dimension {}'.format([1,1], action)
    return action[0]
  else: return action[0]

def obs_formatting(env_tag, obs):
  """
    This function helps reformat the observations according to the environment
    :param env_tag: Environment name
    :param obs: Observation to reformat
    :return:
  """
  if env_tag == 'MountainCarContinuous-v0':
    return np.array([obs])
  elif env_tag == 'Billiard-v0':
    return np.array([np.concatenate(obs)])
  else: return obs
