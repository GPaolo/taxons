from rnd_qd.core.rnd import net
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(7)

def test_layers():
  target = net.TargetNet(6, 3, torch.device("cpu"), fixed=False)
  for name, l in target.named_parameters():
    assert l.requires_grad == True, 'Layer {} is fixed'.format(name)

    target = net.TargetNet(6, 3, torch.device("cpu"), fixed=True)
  for name, l in target.named_parameters():
    assert l.requires_grad == False, 'Layer {} is not fixed'.format(name)


def test_device():
  device = torch.device("cpu")
  target = net.TargetNet(6, 3, device)

  for name, l in target.named_parameters():
    assert not l.is_cuda, 'Layer {} is not on CPU'.format(name)

  if torch.cuda.is_available():
    device = torch.device("cuda")
    target.to(device)
    for name, l in target.named_parameters():
      assert l.is_cuda, 'Layer {} is not on GPU'.format(name)

def test_forward():
  device = torch.device("cpu")
  target = net.TargetNet(6, 3, 5, device)

  x = torch.ones(5, 4, 6) # [batch, ts, input_dim]
  x = target(x, train=True)
  assert all(x.data.numpy().shape == np.array([5, 3])), 'Mismatched shape.'

  x0 = x.data.numpy()[0]
  assert np.allclose(x0, np.array([ 1.0197437, -7.180449, 0.67592883]), rtol=1e-05, atol=1e-05)

def test_backward():
  device = torch.device("cpu")
  target = net.TargetNet(6, 3, 5, device, fixed=False)
  target.zero_grad()
  for name, l in target.named_parameters():
    assert l.grad is None, 'Layer {} grad is not None'.format(name)
  x = torch.ones(5, 4, 6)
  x = target(x)
  criterion = nn.MSELoss()

  loss = criterion(x, torch.ones_like(x))
  loss.backward()
  for name, l in target.named_parameters():
    assert l.grad is not None, 'Layer {} grad is None'.format(name)

