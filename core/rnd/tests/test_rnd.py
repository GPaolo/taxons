from rnd_qd.core.rnd import rnd
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(7)

def test_surprise():
  device = torch.device('cpu')
  net = rnd.RND(6, 2, device)
  x = torch.ones((1,6))
  surprise = net(x)
  # Surprise needs to be different than 0 at the beginning, otherwise we cannot use it.
  assert surprise.item() != 0, 'Surprise value is 0!'

def test_training():
  device = torch.device('cpu')
  net = rnd.RND(6, 2, device)
  x = torch.ones((1, 6))

  loss1 = net.training_step(x)
  loss2 = net.training_step(x)

  assert loss1.item() > loss2.item(), 'Loss does not decreases with training.'



