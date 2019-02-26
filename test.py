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

    # self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=1), nn.ReLU(), # 75 -> 37
    #                              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2), nn.ReLU(), # 37 -> 17
    #                              nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1), nn.ReLU()).cuda(self.device) # 17 -> 8
    #
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3, stride=1), nn.ReLU(), # 8 -> 17
    #                              nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2), nn.ReLU(), # 17 -> 35
    #                              nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=3, stride=2), nn.ReLU()).cuda(self.device) # 35 -> 75

    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, stride=2), nn.ReLU(), # 75 -> 35
                                 nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=3), nn.ReLU()).cuda(self.device) # 35 -> 11

    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=5, stride=3), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=7, stride=2), nn.ReLU()).cuda(self.device)

    self.criterion = nn.MSELoss().cuda(self.device)
    self.learning_rate = 0.01
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
    x = self.subsample(x)
    loss = self.criterion(x, y)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
    self.optimizer.step()
    return loss


if __name__ == '__main__':
  from tensorboardX import SummaryWriter

  import matplotlib.pyplot as plt
  with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
    x = np.load(f)

  writer = SummaryWriter('/home/giuseppe/src/rnd_qd/runs')

  norm = False
  factor = 1
  if norm:
    factor = 255

  net = AutoEncoder()
  x = torch.Tensor(x/factor).permute(0, 3, 1, 2).cuda(net.device)
  test = x[15:16]
  train = x[0:15]

  print('Starting training')
  for k in range(100000):
    loss = net.train_ae(train)
    writer.add_scalar('loss', loss, k)


  # fig, ax = plt.subplots(4, 4)

  writer.export_scalars_to_json("./all_scalars.json")
  writer.close()

  # for i in range(4):
  #   for j in range(4):
  #     k = i+j
  #     b = net(x[k:k+1])
  #     a = b[0]
  #     a = a.permute(1, 2, 0)
  #     a = a.cpu().data.numpy()
  #     if not norm:
  #       a = a.astype(np.int)
  #
  #     ax[i, j].imshow(a)
  #     # plt.imshow(a)
  # plt.show()
  #
  #
  # fig, ax = plt.subplots(1, 2)
  # b = net(test)
  # a = b[0]
  # a = a.permute(1, 2, 0)
  # a = a.cpu().data.numpy()
  # if not norm:
  #   a = a.astype(np.int)
  # ax[0].imshow(a)
  #
  # test = test[0].permute(1,2,0)
  # test = test.cpu().data.numpy()
  # ax[1].imshow(test)
  #
  # k = net.subsample(x[15:16])
  # ll = net.criterion(b, k)
  # print(ll)
  # plt.show()
