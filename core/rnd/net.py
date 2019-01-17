import torch
import torch.nn as nn
import torch.optim as optim

class TargetNet(nn.Module):
  '''
  This class defines the networks used for the RND
  '''
  def __init__(self, input_shape, output_shape, reduced_bs_shape=2, pop_size=10, device=None, fixed=True):
    super(TargetNet, self).__init__()
    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.input_shape = input_shape
    self.output_shape = output_shape
    self.reduced_bs_shape = reduced_bs_shape
    self.hidden_dim = 16
    self.batch_dim = pop_size
    self.fixed = fixed

    self.bn = nn.BatchNorm1d(self.input_shape, track_running_stats=False, affine=False)
    self.lstm = nn.LSTM(self.input_shape, self.hidden_dim)
    self.linear = nn.Sequential(nn.Linear(self.hidden_dim, 32), nn.Tanh(),
                                nn.Linear(32, 64), nn.Tanh(),
                                nn.Linear(64, 128), nn.Tanh(),
                                nn.Linear(128, 32), nn.Tanh(),
                                nn.Linear(32,16), nn.Tanh(),
                                nn.Linear(16, self.output_shape), nn.Tanh())

    self.reduced_bs = nn.Sequential(nn.Linear(self.output_shape, 16), nn.Tanh(),
                                    nn.Linear(16, self.reduced_bs_shape))

    self.linear.apply(self.init_layers)
    self.reduced_bs.apply(self.init_layers)
    self.lstm.apply(self.init_layers)

    self.to(device)
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.zero_grad()

  def init_layers(self, m):
    '''
    Initializes layer m with uniform distribution
    :param m: layer to initialize
    :return:
    '''
    if type(m) == nn.Linear:
      nn.init.normal_(m.weight)
    elif type(m) == nn.LSTM:
      nn.init.normal_(m.weight_ih_l0)
      nn.init.normal_(m.weight_hh_l0)

  def init_hidden(self, train=False):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    # For the target we init the hidden layers as zeros cause every run with the same inputs has to give the same result
    if not train:
      return (torch.zeros(1, 1, self.hidden_dim, device=self.device),
            torch.zeros(1, 1, self.hidden_dim, device=self.device))
    else:
      return (torch.zeros(1, self.batch_dim, self.hidden_dim, device=self.device),
              torch.zeros(1, self.batch_dim, self.hidden_dim, device=self.device))

  def forward(self, x, train=False):
    hidden = self.init_hidden(train)
    x = self.bn(x)
    x = x.transpose(1, 0)
    x, hidden = self.lstm(x, hidden)
    x = self.linear(x[-1])
    r_bs = self.reduced_bs(x)
    return x, r_bs


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
    self.linear = nn.Sequential(nn.Linear(self.hidden_dim, 16), nn.Tanh(),
                                nn.Linear(16, 32), nn.Tanh(),
                                nn.Linear(32, 16), nn.Tanh(),
                                nn.Linear(16, self.output_shape))
    self.linear.apply(self.init_layers)
    self.lstm.apply(self.init_layers)

    self.to(device)
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.zero_grad()

  def init_layers(self, m):
    '''
    Initializes layer m with uniform distribution
    :param m:
    :return:
    '''
    if type(m) == nn.Linear:
      nn.init.normal_(m.weight, mean=0, std=10)
    elif type(m) == nn.LSTM:
      nn.init.normal_(m.weight_ih_l0, mean=0, std=1)
      nn.init.normal_(m.weight_hh_l0, mean=0, std=1)

  def init_hidden(self, train=False):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    if not train:
      return (torch.rand(1, 1, self.hidden_dim, device=self.device),
              torch.rand(1, 1, self.hidden_dim, device=self.device))
    else:
      return (torch.rand(1, self.batch_dim, self.hidden_dim, device=self.device),
              torch.rand(1, self.batch_dim, self.hidden_dim, device=self.device))

  def forward(self, x, train=False):
    hidden = self.init_hidden(train)
    x = self.bn(x)
    x = x.transpose(1, 0)
    x, hidden = self.lstm(x, hidden)
    x = self.linear(x[-1])
    try:
      assert not torch.isnan(x).any()
    except:
      print('Getting NAN.')
      raise
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