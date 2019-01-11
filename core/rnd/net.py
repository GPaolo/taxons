import torch
import torch.nn as nn
import torch.optim as optim

class TargetNet(nn.Module):
  '''
  This class defines the networks used for the RND
  '''
  def __init__(self, input_shape, output_shape, device=None, fixed=False):
    super(TargetNet, self).__init__()
    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.input_shape = input_shape
    self.output_shape = output_shape
    self.fixed = fixed

    self.fc1 = nn.Linear(self.input_shape, 32)
    self.fc2 = nn.Linear(32, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, 32)
    self.fc5 = nn.Linear(32, 16)
    self.fc6 = nn.Linear(16, self.output_shape)

    self.to(device)
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.zero_grad()

  def forward(self, x):
    x = torch.tanh(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    x = torch.tanh(self.fc4(x))
    x = torch.tanh(self.fc5(x))
    x = torch.tanh(self.fc6(x))
    return x


class PredictorNet(nn.Module):
  '''
  This class defines the networks used for the RND
  '''
  def __init__(self, input_shape, output_shape, device=None, fixed=False):
    super(PredictorNet, self).__init__()
    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.input_shape = input_shape
    self.output_shape = output_shape
    self.fixed = fixed

    self.fc1 = nn.Linear(self.input_shape, 16)
    self.fc2 = nn.Linear(16, 32)
    self.fc3 = nn.Linear(32, 32)
    self.fc4 = nn.Linear(32, 16)
    self.fc5 = nn.Linear(16, self.output_shape)

    self.to(device)
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.zero_grad()

  def forward(self, x):
    x = torch.tanh(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    x = torch.tanh(self.fc4(x))
    x = torch.tanh(self.fc5(x))
    # x = torch.tanh(self.fc6(x))
    return x