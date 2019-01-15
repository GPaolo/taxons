import torch
import torch.nn as nn
import torch.optim as optim

class TargetNet(nn.Module):
  '''
  This class defines the networks used for the RND
  '''
  def __init__(self, input_shape, output_shape, pop_size=10, device=None, fixed=True):
    super(TargetNet, self).__init__()
    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.input_shape = input_shape
    self.output_shape = output_shape
    self.hidden_dim = 16
    self.batch_dim = pop_size
    self.fixed = fixed

    self.bn = nn.BatchNorm1d(self.input_shape, track_running_stats=False, affine=False)
    self.lstm = nn.LSTM(self.input_shape, self.hidden_dim)
    self.fc1 = nn.Linear(self.hidden_dim, 32)
    self.fc2 = nn.Linear(32, 64)
    self.fc3 = nn.Linear(64, 128)
    self.fc4 = nn.Linear(128, 32)
    self.fc5 = nn.Linear(32, 16)
    self.fc6 = nn.Linear(16, self.output_shape)

    self.to(device)
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.zero_grad()

  def init_hidden(self, train=False):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    if not train:
      return (torch.rand(1, 1, self.hidden_dim),
            torch.rand(1, 1, self.hidden_dim))
    else:
      return (torch.rand(1, self.batch_dim, self.hidden_dim),
              torch.rand(1, self.batch_dim, self.hidden_dim))

  def forward(self, x, train=False):
    hidden = self.init_hidden(train)

    x = self.bn(x)
    x = x.transpose(1, 0)
    x, hidden = self.lstm(x, hidden)

    x = torch.tanh(self.fc1(x[-1]))
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
  def __init__(self, input_shape, output_shape, pop_size=10, device=None, fixed=False):
    super(PredictorNet, self).__init__()
    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.input_shape = input_shape
    self.output_shape = output_shape
    self.hidden_dim = 16
    self.batch_dim = pop_size
    self.fixed = fixed

    self.bn = nn.BatchNorm1d(self.input_shape, track_running_stats=False, affine=False)
    self.lstm = nn.LSTM(self.input_shape, self.hidden_dim)
    self.fc1 = nn.Linear(self.hidden_dim, 16)
    self.fc2 = nn.Linear(16, 32)
    self.fc3 = nn.Linear(32, 16)
    self.fc4 = nn.Linear(16, self.output_shape)

    self.to(device)
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.zero_grad()

  def init_hidden(self, train=False):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    if not train:
      return (torch.rand(1, 1, self.hidden_dim),
            torch.rand(1, 1, self.hidden_dim))
    else:
      return (torch.rand(1, self.batch_dim, self.hidden_dim),
              torch.rand(1, self.batch_dim, self.hidden_dim))

  def forward(self, x, train=False):
    hidden = self.init_hidden(train)

    x = self.bn(x)
    x = x.transpose(1, 0)
    x, hidden = self.lstm(x, hidden)

    x = torch.tanh(self.fc1(x[-1]))
    x = torch.tanh(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    x = self.fc4(x)
    return x

if __name__ == '__main__':
  net = PredictorNet(4, 3, pop_size=10)
  nett = TargetNet(4, 3, pop_size=10)
  t = torch.load('../../test_tensor.pt')
  print(t.size())
  a = net(t)
  b = nett(t)
  print(a)
  print(b)
  criterion = nn.MSELoss(reduction='none')
  c = criterion(a,b).cpu().data.numpy()
  import numpy as np
  print(np.shape(c))
  print(np.sum(c, axis=1))