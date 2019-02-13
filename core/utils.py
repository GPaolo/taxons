import numpy as np
import matplotlib.pyplot as plt
import os

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


class DMP(object):
  def __init__(self, num_bf=20):
    self.num_bf = num_bf
    self.mu = np.abs(np.random.randn(self.num_bf))
    self.sigma = np.random.uniform(size=self.num_bf)
    self.w = np.random.randn(self.num_bf)
    self.a_x  = np.random.uniform()
    self.tau = 100

  def __call__(self, t):
    g = np.zeros(self.num_bf)
    x = np.exp( -self.a_x*t/self.tau )

    for k in range(self.num_bf):
      g[k] = self.basis_function(x, self.mu[k], self.sigma[k])
    f = np.sum(g*self.w)/np.sum(g)*x
    return f

  def basis_function(self, x, mu, sigma):
    """
    This basis function samples the value of gaussian in x.
    :param x: Point where to sample the gaussian
    :param mu: Center of the gaussian
    :param sigma: Std deviation
    :return:
    """
    return np.exp(-np.power((x - mu)/sigma, 2)/2)




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

def show(bs_points, name=None):
  print('Behaviour space coverage representation.')

  pts = ([x[0] for x in bs_points if x is not None], [y[1] for y in bs_points if y is not None])
  plt.rcParams["patch.force_edgecolor"] = True
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
  axes[0].set_title('Ball position')
  axes[0].scatter(pts[0], pts[1])
  axes[0].set_xlim(-1.5, 1.5)
  axes[0].set_ylim(-1.5, 1.5)

  axes[1].set_title('Histogram')
  H, xedges, yedges = np.histogram2d(pts[0], pts[1], bins=(100, 100), range=np.array([[-1.5, 1.5], [-1.5, 1.5]]))
  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
  cax = axes[1].matshow(np.rot90(H, k=1), extent=extent)
  axes[1].set_xlim(-1.5, 1.5)
  axes[1].set_ylim(-1.5, 1.5)
  plt.colorbar(cax, ax=axes[1])

  if name is None:
    plt.savefig('./behaviour.pdf')
  else:
    plt.savefig('./{}.pdf'.format(name))
  print('Plot saved in {}/'.format(os.getcwd()))

