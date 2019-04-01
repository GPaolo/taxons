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
import mujoco_py

if __name__ == '__main__':

  env_tag = 'Ant-v2'
  env = gym.make(env_tag)

  env.reset()
  print(env.action_space.sample())
  print(np.array(env.action_space.sample()).shape)
  for _ in range(100):
    env.render()
    a =env.action_space.sample()
    o = env.step(a)  # take a random action
    print(o[0])
    print(np.array(o[0]).shape)
    print('')
  env.close()
  #
  # seed = 2
  # env.seed(seed)
  # np.random.seed(seed)
  # torch.manual_seed(seed)
