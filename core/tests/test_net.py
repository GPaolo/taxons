from rnd_qd.core import rnd
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(7)

def test_layers():
  net = rnd.Net(6, 3, torch.device("cpu"), fixed=False)
  for name, l in net.named_parameters():
    assert l.requires_grad == True, 'Layer {} is fixed'.format(name)

  net = rnd.Net(6, 3, torch.device("cpu"), fixed=True)
  for name, l in net.named_parameters():
    assert l.requires_grad == False, 'Layer {} is not fixed'.format(name)


def test_device():
  device = torch.device("cpu")
  net = rnd.Net(6, 3, device)

  for name, l in net.named_parameters():
    assert not l.is_cuda, 'Layer {} is not on CPU'.format(name)

  if torch.cuda.is_available():
    device = torch.device("cuda")
    net.to(device)
    for name, l in net.named_parameters():
      assert l.is_cuda, 'Layer {} is not on GPU'.format(name)

def test_forward():
  device = torch.device("cpu")
  net = rnd.Net(6, 3, device)

  x = torch.ones(6)
  x = net(x)
  assert x.shape[0] == 3, 'Mismatched shape.'
  assert np.allclose(x.data.numpy(), np.array([ 0.1881, -0.1422,  0.0317]), rtol=1e-04, atol=1e-04)

def test_backward():
  device = torch.device("cpu")
  net = rnd.Net(6, 3, device, fixed=False)
  net.zero_grad()
  for name, l in net.named_parameters():
    assert l.grad is None, 'Layer {} grad is not None'.format(name)
  x = torch.ones(6)
  x = net(x)
  criterion = nn.MSELoss()

  loss = criterion(x, torch.ones_like(x))
  loss.backward()
  for name, l in net.named_parameters():
    assert l.grad is not None, 'Layer {} grad is None'.format(name)

