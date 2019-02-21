# Created by Giuseppe Paolo 
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AutoEncoder(nn.Module):

  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2), nn.ReLU(),
                                 nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2), nn.ReLU(),
                                 nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3), nn.ReLU())

    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=6, stride=2), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=6, stride=2), nn.ReLU())

    self.criterion = nn.MSELoss()
    self.zero_grad()
    self.optimizer = optim.SGD(self.parameters(), 0.0001)

  def forward(self, x):
    y = self.encoder(x)
    y = self.decoder(y)
    return y

  def train_ae(self, x):
    self.optimizer.zero_grad()
    y = self.forward(x)
    loss = self.criterion(x, y)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), 0.1)
    self.optimizer.step()
    print(loss)


if __name__ == '__main__':
  x = torch.Tensor(np.random.random((1, 3, 100, 100)))
  net = AutoEncoder()

  for k in range(1000):
    net.train_ae(x)



  print(loss)