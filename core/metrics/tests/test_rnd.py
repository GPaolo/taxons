from core.rnd_qd import rnd
import torch
import numpy as np

torch.manual_seed(7)
np.random.seed(7)
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

examples = 1

def test_base_net():
  encoding_shape = 3
  net = rnd.BaseNet(encoding_shape)
  net.to(device)
  x = torch.Tensor(np.ones((examples, 3, 64, 64))).to(device)
  out = net(x)
  assert np.all(out.shape == np.array([examples, encoding_shape]))

def test_subsample():
  encoding_shape = 3
  net = rnd.RND(encoding_shape, device=device)
  x = torch.Tensor(np.ones((examples, 3, 300, 300))).to(device)
  y = net.subsample(x)
  assert np.all(y.shape == np.array([examples, 3, 64, 64]))

def test_surprise():
  encoding_shape = 3
  net = rnd.RND(encoding_shape, device=device)
  x = torch.Tensor(np.ones((examples, 3, 300, 300))).to(device)
  surprise = net(x)
  print(surprise)
  # Surprise needs to be different than 0 at the beginning, otherwise we cannot use it.
  assert np.all(surprise[0].cpu().data.numpy() != np.zeros(examples)), 'Surprise value is 0!'
  assert np.all(surprise[1].shape == np.array([examples, encoding_shape]))

def test_training():
  encoding_shape = 3
  net = rnd.RND(encoding_shape, device=device)
  x = torch.Tensor(np.ones((examples, 3, 300, 300))).to(device)

  loss1 = net.training_step(x)
  loss2 = net.training_step(x)

  assert loss1[0].cpu().data.numpy() > loss2[0].cpu().data.numpy(), 'Loss does not decreases with training.'
  assert np.all(loss1[1].shape == np.array([examples, encoding_shape]))



