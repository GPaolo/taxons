import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from torch.optim.lr_scheduler import _LRScheduler


# ---------------------------------------------------------------------------
class LRScheduler(_LRScheduler):
  """
  Scales the LR of a given factor every new metric update cycle
  """

  # ---------------------------------------------------
  def __init__(self, optimizer, scale, last_epoch=0):
    """
    The LR scheduler for the LR of the nets
    :param optimizer: Optimizer of the nets
    :param scale: The scaling factor
    :param last_epoch: The last epoch
    """
    self.optimizer = optimizer
    self.last_epoch = last_epoch
    self.scale = scale
    self.init_call = True
    super(LRScheduler, self).__init__(self.optimizer)
  # ---------------------------------------------------

  # ---------------------------------------------------
  def get_lr(self):
    """
    Get current LR
    :return: LR
    """
    if self.init_call:
      self.init_call = False
      return [group['lr'] for group in self.optimizer.param_groups]

    lr = [group['lr'] * self.scale for group in self.optimizer.param_groups]
    print("New lr: {}".format(lr))
    return lr
  # ---------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class Logger(object):
  """
  This class works as a logger for the experiments
  """
  def __init__(self, log_dict=None):
    """
    Constructor
    :param log_dict: This is the dict that will be used to log
    """
    if log_dict is None:
      self.log = {}
    else:
      self.log = log_dict

  def register_log(self, key, value):
    """
    This function adds another value to the log
    :param key: Which key of the log dict add the new value to
    :param value: What value to add
    """
    self.log[key].append(str(value))

  def save(self, filepath):
    """
    This function saves the logs to the given filepath
    :param filepath:
    :return:
    """
    cwd = os.getcwd()
    try:
      assert os.path.exists(filepath), 'Specified path for logs does not exists. Saving in {}'.format(cwd)
    except:
      filepath = cwd

    with open(os.path.join(filepath, 'logs.json'), 'w') as f:
      json.dump(self.log, f, indent=4)
# ---------------------------------------------------------------------------


# ---------------------------------------------------
def action_formatting(env_tag, action):
  """
  This function helps reformat the actions according to the environment
  :param env_tag: Environment name
  :param action: Action to reformat
  :return: The formatted action
  """
  if env_tag == 'MountainCarContinuous-v0':
    assert action.shape == (1,1), 'Shape is not of dimension {}. Has dimension {}'.format([1,1], action)
    return action[0]
  elif 'Fastsim' in env_tag:
    return action[0]*5 # The NN generates actions in the [-1, 1] (tanh), we scale it to the max range of actions of the env [-5, 5]
  else:
    return action[0]
# ---------------------------------------------------


# ---------------------------------------------------
def extact_hd_bs(env, obs, reward=None, done=None, info=None):
  """
    This function helps extract the hand designed BS used to compare the approaches
    :param env_tag: Environment name
    :param obs: Observation to reformat
    :return: The extracted ground truth BS
  """
  env_tag = env.spec.id
  if env_tag == 'MountainCarContinuous-v0':
    return np.array(obs)
  elif env_tag == 'Billiard-v0':
    return np.array([obs[0][0], obs[0][1]])
  elif env_tag == 'BilliardHard-v0':
    return np.array([obs[0][0], obs[0][1]])
  elif env_tag == 'AntMuJoCoEnv-v0':
    return np.array(env.robot.body_xyz[:2]) # xy position of CoM of the robot
  elif env_tag == 'Ant-v2':
    return np.array(env.env.data.qpos[:2])
  elif env_tag == 'FastsimSimpleNavigation-v0':
    if info is None:
      return None
    return np.array(info['robot_pos'][:2])
  else:
    return obs
# ---------------------------------------------------


# ---------------------------------------------------
def get_projectpath():
  """
  Finds the projectpact
  :return: Absolute path of the project
  """
  cwd = os.getcwd()
  folder = os.path.basename(cwd)
  while not folder == 'taxons':
    cwd = os.path.dirname(cwd)
    folder = os.path.basename(cwd)
  return cwd
# ---------------------------------------------------


# ---------------------------------------------------
def load_maze_image():
  """
  Loads the image of the maze for the Fastsim simulator
  :return: The image of the maze as numpy array
  """
  import netpbmfile as npbm
  path = os.path.join(get_projectpath(), 'external/fastsim_gym/gym_fastsim/simple_nav/assets/maze_hard.pbm')
  with open(path, 'rb') as f:
    maze = np.array(npbm.imread(f))
  return maze
# ---------------------------------------------------


# ---------------------------------------------------
def show(bs_points, filepath, name=None, info=None, upper_limit=1.35, lower_limit=-1.35):
  """
  Shows the coverage of the found solutions in the ground truth space
  :param bs_points: BS points in the ground truth space
  :param filepath: Path where to save the image
  :param name: Name of the plot
  :param info: Info about the data
  :param upper_limit: Upper limit of the graph
  :param lower_limit: Bottom limit of the graph
  :return: The coverage value
  """
  print('Seed {} - Behaviour space coverage representation.'.format(info['seed']))
  pts = ([x[0] for x in bs_points if x is not None], [y[1] for y in bs_points if y is not None])
  plt.rcParams["patch.force_edgecolor"] = True
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
  axes[0].set_title('Final position')
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
# ---------------------------------------------------


# ---------------------------------------------------
def split_array(a, batch_size=32, shuffle=True):
  """
  Splits a data array in arrays of the given batch size
  :param a: Array to split
  :param batch_size: Batch size of new arrays
  :param shuffle: Flag to shuffle the array
  :return: A list of arrays of batch_size
  """
  length = len(a)
  parts = int(np.ceil(length/batch_size))
  if shuffle:
    np.random.shuffle(a)
  return [a[k*batch_size:min(length, (k+1)*batch_size)] for k in range(parts)]
# ---------------------------------------------------
