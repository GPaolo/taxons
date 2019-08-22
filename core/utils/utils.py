import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import traceback
from torch.optim.lr_scheduler import _LRScheduler

class FCLayer(object):

  def __init__(self, input, output, name='fc', bias=True):
    super().__init__()
    self.w = np.random.randn(input, output)
    if bias:
      self.bias = np.random.randn(1, output)
    else:
      self.bias = np.zeros((1, output))
    self.name = name

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    assert np.shape(x)[0] == 1, 'Wrong input shape. Needs to be [1, n]. Is {}'.format(np.shape(x))
    return np.matmul(x, self.w) + self.bias

  @property
  def params(self):
    return {'w': self.w, 'bias': self.bias}

  @property
  def show(self):
    print('Layer {}:'.format(self.name))
    print('Weights {}'.format(self.w))
    print('Bias {}'.format(self.bias))


class DMPExp(object):
  def __init__(self, name='dmp', **kwargs):
    self.num_basis_func = kwargs['degree']
    self.mu = np.abs(np.random.randn(self.num_basis_func))
    self.sigma = np.random.uniform(size=self.num_basis_func)
    self.w = np.random.randn(self.num_basis_func)
    self.a_x  = np.random.uniform()
    self.tau = 2
    self.name = name

  def __call__(self, t):
    g = np.zeros(self.num_basis_func)
    x = np.exp( -self.a_x*t/self.tau )

    for k in range(self.num_basis_func):
      g[k] = self.basis_function(x, self.mu[k], self.sigma[k])
    f = np.sum(g*self.w)/np.sum(g)
    return f

  @property
  def params(self):
    return {'mu': self.mu, 'sigma': self.sigma, 'w': self.w, 'a_x': self.a_x, 'tau': self.tau, 'name':self.name}

  @staticmethod
  def basis_function(x, mu, sigma):
    """
    This basis function samples the value of gaussian in x.
    :param x: Point where to sample the gaussian
    :param mu: Center of the gaussian
    :param sigma: Std deviation
    :return:
    """
    return np.exp(np.sin((x - mu)/sigma)/2)

  def load(self, params):
    self.mu = params['mu']
    self.sigma = params['sigma']
    self.w = params['w']
    self.a_x = params['a_x']
    self.tau = params['tau']
    self.name = params['name']


class DMPPoly(object):
  def __init__(self, name='dmp', **kwargs):
    self.degree = kwargs['degree']
    self.w = np.random.randn(self.degree+1)
    self.scale = 1
    self.name = name

  def __call__(self, t):
    x = np.cos(t/self.scale) # This way we limit it between [-1, 1]
    p = 0
    for i in range(self.degree+1):
      p += self.w[i]*x**i
    return p

  @property
  def params(self):
    return {'w': self.w, 'scale':self.scale, 'name':self.name}

  def load(self, params):
    self.scale = params['scale']
    self.w = params['w']
    self.name = params['name']


class DMPSin(object):
  def __init__(self, name='dmp', **kwargs):
    self.name = name
    self.amplitude = np.random.randn()
    self.period = np.random.uniform()

  def __call__(self, t):
    x = self.amplitude * np.sin(2*np.pi*t/self.period)
    return x

  @property
  def params(self):
    return {'period': self.period, 'amplitude': self.amplitude, 'name':self.name}

  @property
  def period(self):
    return self.__period

  @period.setter
  def period(self, x):
    self.__period = x

  @property
  def amplitude(self):
    return self.__amplitude

  @amplitude.setter
  def amplitude(self, x):
    if x > 5:
      self.__amplitude = 5
    elif x < -5:
      self.__amplitude = -5
    else:
      self.__amplitude = x

  def load(self, params):
    self.period = params['period']
    self.amplitude = params['amplitude']
    self.name = params['name']


class LRScheduler(_LRScheduler):
  """
  Scales the LR of a given factor every new metric update cycle
  """
  def __init__(self, optimizer, scale, last_epoch=0):
    self.optimizer = optimizer
    self.last_epoch = last_epoch
    self.scale = scale
    self.init_call = True
    super(LRScheduler, self).__init__(self.optimizer)

  def get_lr(self):
    if self.init_call:
      self.init_call = False
      return [group['lr'] for group in self.optimizer.param_groups]

    lr = [group['lr'] * self.scale for group in self.optimizer.param_groups]
    print("New lr: {}".format(lr))
    return lr


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
  elif 'Fastsim' in env_tag:
    return action[0]*5 # The NN generates actions in the [-1, 1] (tanh), we scale it to the max range of actions of the env [-5, 5]
  else:
    return action[0]


def extact_hd_bs(env, obs, reward=None, done=None, info=None):
  """
    This function helps extract the hand designed BS used to compare the approaches
    :param env_tag: Environment name
    :param obs: Observation to reformat
    :return:
  """
  env_tag = env.spec.id
  if env_tag == 'MountainCarContinuous-v0':
    return np.array([obs])
  elif env_tag == 'Billiard-v0':
    return np.array([[obs[0][0], obs[0][1]]])
  elif env_tag == 'BilliardHard-v0':
    return np.array([[obs[0][0], obs[0][1]]])
  elif env_tag == 'Ant-v2':
    return np.array([env.robot.body_xyz[:2]]) # xy position of CoM of the robot
  elif env_tag == 'FastsimSimpleNavigation-v0':
    if info is None:
      return None
    return np.array([info['robot_pos'][:2]])
  else:
    return obs


def get_projectpath():
  cwd = os.getcwd()
  folder = os.path.basename(cwd)
  while not folder == 'rnd_qd':
    cwd = os.path.dirname(cwd)
    folder = os.path.basename(cwd)
  return cwd


def load_maze_image():
  import netpbmfile as npbm
  path = os.path.join(get_projectpath(), 'external/fastsim_gym/gym_fastsim/simple_nav/assets/maze_hard.pbm')
  with open(path, 'rb') as f:
    maze = np.array(npbm.imread(f))
  return maze


def show(bs_points, filepath, name=None, info=None, upper_limit=1.35, lower_limit=-1.35):

  # maze = None
  # if 'maze' in filepath or 'Maze' in filepath:
  #   maze = load_maze_image()

  print('Seed {} - Behaviour space coverage representation.'.format(info['seed']))
  pts = ([x[0] for x in bs_points if x is not None], [y[1] for y in bs_points if y is not None])
  plt.rcParams["patch.force_edgecolor"] = True
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
  axes[0].set_title('Final position')
  # if maze is not None:
  #   axes[0].imshow(maze)
  axes[0].scatter(pts[0], pts[1])
  axes[0].set_xlim(lower_limit, upper_limit)
  axes[0].set_ylim(lower_limit, upper_limit)

  axes[1].set_title('Histogram')
  H, xedges, yedges = np.histogram2d(pts[0], pts[1], bins=(50, 50), range=np.array([[lower_limit, upper_limit], [lower_limit, upper_limit]]))
  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
  cax = axes[1].matshow(np.rot90(H, k=1), extent=extent)
  axes[1].set_xlim(lower_limit, upper_limit)
  axes[1].set_ylim(lower_limit, upper_limit)
  plt.colorbar(cax, ax=axes[1])

  coverage = np.count_nonzero(H)/(50*50)*100
  if name is None:
    fig.suptitle("Generation {} - Coverage {}%\n".format(info['gen'], coverage), fontsize=16)
    plt.savefig(os.path.join(filepath, 'behaviour.pdf'))
  else:
    with open(os.path.join(filepath, 'data.txt'), 'a+') as f:
      f.write("Coverage {}%\n".format(coverage))
      f.write("Total solutions found: {}\n".format(len(bs_points)))
      if info is not None:
        inf = json.dumps(info)
        f.write(inf)

    plt.savefig(os.path.join(filepath, '{}.pdf'.format(name)))
    print('Seed {} - Plot saved in {}'.format(info['seed'], filepath))
  plt.close(fig)
  return coverage


def split_array(a, batch_size=32, shuffle=True):
  length = len(a)
  parts = int(np.ceil(length/batch_size))
  if shuffle:
    np.random.shuffle(a)
  return [a[k*batch_size:min(length, (k+1)*batch_size)] for k in range(parts)]


def rgb2gray(img):
  gray = 0.2989 * img[:,:,:,0] + 0.5870 * img[:,:,:,1] + 0.1140 * img[:,:,:,2]
  return np.expand_dims(gray, -1)


def calc_overlapping(grid_size, archive, coverage):
  total_area = grid_size[0] * grid_size[1]
  occupied_cells = total_area * coverage/100
  overlapping = 1 - occupied_cells/archive
  return overlapping*100



