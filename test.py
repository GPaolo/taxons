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
import mujoco_py
import matplotlib.pyplot as plt
# os.environ['LD_LIBRARY_PATH'] = '/home/giuseppe/.mujoco/mujoco200/bin:/usr/lib/nvidia-410'
# os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so'
import matplotlib
from core.metrics import ae


if __name__ == '__main__':
  os.unsetenv('LD_PRELOAD')

  env = gym.make('Ant-v3')
  state = env.reset()
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  print(obs_dim)
  print(act_dim)
  for i in range(0):
    action = np.zeros_like(env.action_space.sample())
    _, _, _, info = env.step(action)
    print("x: {} - y: {}".format(info['x_position'], info['y_position']))
  state = np.array(env.render(mode='rgb_array'))
  depth= np.array(env.render(mode='depth_array'))
  #depth = 1- np.expand_dims(depth, -1)
  #depth = depth/np.max(depth)
  #state = depth * state/255
  print(state.shape)
  plt.imshow(state)
  plt.show()
  # matplotlib.image.imsave('mujoco_image.png', state)
  #

  # with open('/home/giuseppe/src/rnd_qd/mujoco_image.png', 'rb') as ff:
  #   state = matplotlib.pyplot.imread(ff)
  state = state[:, :, :3]
  #plt.imshow(depth)
  #plt.show()
  print(state.shape)
  encod = ae.ConvAutoEncoder(encoding_shape=16, learning_rate=0.0001)
  state = torch.Tensor(state).permute(2, 0, 1).unsqueeze(0)

  for k in range(5000):
    loss ,_ = encod.training_step(state)
    print('{}: {}'.format(k, loss.data))
  y, _ = encod.forward(state)
  print(y.shape)
  y = y[0].permute(1, 2, 0)
  print(y.data.numpy().shape)
  plt.imshow(y.data.numpy()/255)
  plt.show()
