# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from core.rnd.net import TargetNet, PredictorNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RND(object):
  '''
  Class that instantiates the RND component
  '''
  def __init__(self, input_shape, encoding_shape, device=None):
    self.input_shape = input_shape
    self.encoding_shape = encoding_shape
    self.device = device
    # Nets
    self.target_model = TargetNet(self.input_shape, self.encoding_shape, self.device, fixed=True)
    self.predictor_model = PredictorNet(self.input_shape, self.encoding_shape, self.device, fixed=False)
    # Loss
    self.criterion = nn.MSELoss(reduction='sum')
    self.training_criterion = nn.MSELoss()
    # Optimizer
    self.learning_rate = 0.00001
    self.optimizer = optim.SGD(self.predictor_model.parameters(), self.learning_rate)

  def _get_surprise(self, x, train=False):
    '''
    This function calculates the surprise given by the input
    :param x: Network input. Needs to be a torch tensor.
    :return: surprise as a 1 dimensional torch tensor
    '''
    target = self.target_model(x)
    prediction = self.predictor_model(x)
    if not train:
      return self.criterion(prediction, target)
    else:
      return self.criterion(prediction, target)/10

  def __call__(self, x):
    return self._get_surprise(x)

  def training_step(self, x):
    '''
    This function performs the training step.
    :param x: Network input. Needs to be a torch tensor.
    :return: surprise as a 1 dimensional torch tensor
    '''
    self.optimizer.zero_grad()
    surprise = self._get_surprise(x, train=True)
    surprise.backward()
    self.optimizer.step()
    return surprise




if __name__ == '__main__':
  device = torch.device('cpu')
  rnd = RND(6, 2, device)

  for _ in range(1000):
    x = torch.rand([1, 6])
    print(rnd.training_step(x))

  x = torch.Tensor([0.2, 0.4, 0.557, 0.36, 0, 1.5])
  print(rnd(x))

