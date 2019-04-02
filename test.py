# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import gym_billiard
import sys
from core.qd import population, agents
from core.utils import utils
import os

# os.environ['LD_LIBRARY_PATH'] = '/home/giuseppe/.mujoco/mujoco200/bin:/usr/lib/nvidia-410'
# os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so'

if __name__ == '__main__':

  d = {'a':3}
  aa = utils.DMPSin(*d)
  print(aa.amplitude)
  print(aa.params)
