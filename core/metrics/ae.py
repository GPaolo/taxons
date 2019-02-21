# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class AutoEncoder(nn.Module):

  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.subsample = nn.AvgPool2d(8).cuda(self.device) # 600 -> 75

    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=1), nn.ReLU(), # 75 -> 37
                                 nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2), nn.ReLU(), # 37 -> 17
                                 nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1), nn.ReLU()).cuda(self.device) # 17 -> 8

    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3, stride=1), nn.ReLU(), # 8 -> 17
                                 nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2), nn.ReLU(), # 17 -> 35
                                 nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=3, stride=2), nn.ReLU()).cuda(self.device) # 35 -> 75

    # self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, stride=2), nn.ReLU(), # 75 -> 35
    #                              nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=3), nn.ReLU()).cuda(self.device) # 35 -> 11
    #
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=5, stride=3), nn.ReLU(),
    #                              nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=7, stride=2), nn.ReLU()).cuda(self.device)

    self.criterion = nn.MSELoss().cuda(self.device)
    self.learning_rate = 0.01
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)

  def _get_surprise(self, x):
    y = self.forward(x)
    x = self.subsample(x)
    loss = self.criterion(x, y)
    return loss

  def forward(self, x):
    y = self.subsample(x)
    y = self.encoder(y)
    y = self.decoder(y)
    return y

  def training_step(self, x):
    self.optimizer.zero_grad()
    novelty = self._get_surprise(x)
    novelty.backward()
    self.optimizer.step()
    return novelty

  def __call__(self, x):
    return self._get_surprise(x)

  def save(self, filepath):
    save_ckpt = {
      'ae': self.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    try:
      torch.save(save_ckpt, os.path.join(filepath, 'ckpt_ae.pth'))
    except:
      print('Cannot save autoencoder.')