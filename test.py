# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import gym_billiard

class AutoEncoder(nn.Module):

  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.subsample = nn.AvgPool2d(8).cuda(self.device) # 600 -> 75

    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=1), nn.ReLU(), # 75 -> 37
                                 nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2), nn.ReLU(), # 37 -> 17
                                 nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1), nn.ReLU()).cuda(self.device) # 17 -> 8

    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=1), nn.ReLU(), # 8 -> 17
                                 nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2), nn.ReLU(), # 17 -> 35
                                 nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, stride=2), nn.ReLU()).cuda(self.device) # 35 -> 75

    self.criterion = nn.MSELoss().cuda(self.device)
    self.learning_rate = 0.001
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)


  def forward(self, x):
    y = self.subsample(x)
    y = self.encoder(y)
    y = self.decoder(y)

    return y

  def train_ae(self, x):
    self.optimizer.zero_grad()

    y = self.forward(x)
    x = self.subsample(x/255)
    loss = self.criterion(x, y)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
    self.optimizer.step()
    print(loss)


if __name__ == '__main__':
  # env = gym.make('Billiard-v0')
  # state = []
  #
  # import matplotlib
  # matplotlib.use('agg')
  # for i in range(16):
  #   env.reset()
  #   state.append(env.render(rendered=False))
  #
  # state = np.array(state)
  #

  import matplotlib.pyplot as plt
  with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
    x = np.load(f)

  # plt.imshow(x/255)
  # plt.show()

  net = AutoEncoder()
  x = torch.Tensor(x).permute(0, 3, 1, 2).cuda(net.device)
  net(x)

  for k in range(5000):
    net.train_ae(x)

  b = net(x[0:1])




  # print(loss)
  a = b[0]
  a = a.permute(1, 2, 0)
  a = a.cpu().data.numpy()


  plt.imshow(a)
  plt.show()