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

    self.subsample = nn.AvgPool3d((1, 8, 8)).to(self.device) # 600 -> 75

    self.encoder = nn.Sequential(nn.Conv3d(in_channels=3, out_channels=8, kernel_size=5, stride=2), nn.ReLU(), # 75 -> 36
                                nn.Conv3d(in_channels=8, out_channels=8, kernel_size=4, stride=2), nn.ReLU(), # 36 -> 17
                                nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1), nn.ReLU()).to(self.device) # 17 -> 15

    self.decoder = nn.Sequential(nn.ConvTranspose3d(in_channels=4, out_channels=8, kernel_size=3, stride=1), nn.ReLU(), # 8 -> 17
                                nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=4, stride=2), nn.ReLU(), # 17 -> 36
                                nn.ConvTranspose3d(in_channels=8, out_channels=3, kernel_size=5, stride=2), nn.ReLU()).to(self.device) # 36 -> 75
    #
    # self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2), nn.LeakyReLU(), # 75 -> 35
    #                              nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=3), nn.LeakyReLU()).to(self.device)  # 35 -> 11
    #
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=5, stride=3), nn.LeakyReLU(),
    #                              nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=7, stride=2), nn.ReLU()).to(self.device)

    # self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, stride=1), nn.ReLU()).to(self.device)  # 75 -> 36
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=7, stride=1), nn.ReLU()).to(self.device)

    # self.encoder = nn.Sequential(nn.Linear(16875, 4096), nn.ReLU(),
    #                              nn.Linear(4096, 1024), nn.ReLU(),
    #                              nn.Linear(1024, 256), nn.ReLU()).to(self.device)
    # self.decoder = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(),
    #                              nn.Linear(1024, 4096), nn.ReLU(),
    #                              nn.Linear(4096, 16875), nn.ReLU()).to(self.device)


    self.zero_grad()
    self.learning_rate = 0.001
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [3000, 8000], 0.1)

    self.criterion = nn.MSELoss().to(self.device)
    self.to(self.device)


  def forward(self, x):
    y = self.subsample(x)
    shape = y.shape
    # y = self.encoder(y.view(shape[0], -1))
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
    self.scheduler.step()
    return loss


if __name__ == '__main__':
  env = gym.make('Billiard-v0')
  #
  env.reset()
  #state = env.render(rendered=False)
  #
  #with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
  #  tens = np.load(f)
  #
  #test = np.append(tens, np.expand_dims(state, 0), 0)
  #with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'wb') as f:
  #  test.dump(f)
  #print(test.shape)

  #from tensorboardX import SummaryWriter

  #import matplotlib.pyplot as plt
  with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
    x = np.load(f)
  print(x.shape)
  #writer = SummaryWriter('/home/giuseppe/src/rnd_qd/runs')

  norm = False
  factor = 1
  if norm:
    factor = 255

  net = AutoEncoder()
  x = torch.Tensor(x/factor).permute(3, 0, 1, 2).to(net.device)
  test = x[:, 15:]
  train = x[:, :15]
  print(test.shape)
  test.to(net.device)
  train.to(net.device)

  #a = net(x.unsqueeze(0))
  print('Starting training')
  for k in range(5000):
    loss = net.train_ae(train.unsqueeze(0))
    if k%100 == 0:
      print('Loss at {}: {}'.format(k, loss))
  #  writer.add_scalar('loss', loss, k)


  #fig, ax = plt.subplots(4, 4)

  #writer.export_scalars_to_json("./all_scalars.json")
  #writer.close()

  #for i in range(4):
   # for j in range(4):
  #    k = i+j
  #    b = net(x[k:k+1])
  #    a = b[0]
  #    a = a.permute(1, 2, 0)
  #    a = a.cpu().data.numpy()
  #    if not norm:
  #      a = a.astype(np.int)

  #    ax[i, j].imshow(a)
      # plt.imshow(a)
  #plt.show()

  #fig, ax = plt.subplots(1, 2)
  #b = net(test)
  #a = b[0]
  #a = a.permute(1, 2, 0)
  #a = a.cpu().data.numpy()
  #if not norm:
  #  a = a.astype(np.int)
  #ax[0].imshow(a)
  #
  #test = test[0].permute(1,2,0)
  #test = test.cpu().data.numpy()
  #ax[1].imshow(test)
  #
  #k = net.subsample(x[15:16])
  #ll = net.criterion(b, k)
  #print(ll)
  #plt.show()
