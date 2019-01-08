# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
  '''
  This class defines the networks used for the RND
  '''
  def __init__(self, input_shape, output_shape, device, fixed=False):
    super(Net, self).__init__()
    self.device = device
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.fixed = fixed

    self.fc1 = nn.Linear(self.input_shape, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 16)
    self.fc4 = nn.Linear(16, self.output_shape)

    self.to(device)
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.zero_grad()

  def forward(self, x):
    x = torch.tanh(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    x = torch.tanh(self.fc4(x))
    return x


class RND(object):
  '''
  Class that instantiates the RND component
  '''
  def __init__(self, input_shape, encoding_shape, device):
    self.input_shape = input_shape
    self.encoding_shape = encoding_shape
    self.device = device
    # Nets
    self.target_model = Net(self.input_shape, self.encoding_shape, self.device, fixed=True)
    self.predictor_model = Net(self.input_shape, self.encoding_shape, self.device, fixed=False)
    # Loss
    self.criterion = nn.MSELoss()
    # Optimizer
    self.learning_rate = 0.01
    self.optimizer = optim.Adam(self.predictor_model.parameters(), self.learning_rate)

  def _get_surprise(self, x):
    '''
    This function calculates the surprise given by the input
    :param x: Network input. Needs to be a torch tensor.
    :return: Surprise
    '''
    target = self.target_model(x)
    prediction = self.predictor_model(x)
    surprise = self.criterion(prediction, target)
    return surprise

  def __call__(self, x):
    return self._get_surprise(x)

  def training_step(self, x):
    '''
    This function performs the training step.
    :param x: Network input. Needs to be a torch tensor.
    :return:
    '''
    self.optimizer.zero_grad()
    surprise = self._get_surprise(x)
    surprise.backward()
    self.optimizer.step()
    return surprise




if __name__ == '__main__':
  device = torch.device('cpu')
  rnd = RND(6, 2, device)
  x = torch.ones(6)

  for _ in range(100):
    print(rnd.training_step(x))

  x = torch.tensor([0.5, 2, 5, 0.2, 3, 1])
  print(rnd(x))

