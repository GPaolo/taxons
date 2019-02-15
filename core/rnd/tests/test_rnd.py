from core.rnd_qd import rnd
import torch

torch.manual_seed(7)

def test_surprise():
  device = torch.device('cpu')
  net = rnd.RND(6, 2, 5, device)
  x = torch.ones(1, 3, 6)
  surprise = net(x)
  # Surprise needs to be different than 0 at the beginning, otherwise we cannot use it.
  assert surprise.item() != 0, 'Surprise value is 0!'

def test_training():
  device = torch.device('cpu')
  net = rnd.RND(6, 2, 5, device)
  x = torch.ones(5, 3, 6)

  loss1 = net.training_step(x)
  loss2 = net.training_step(x)

  assert loss1.item() > loss2.item(), 'Loss does not decreases with training.'



