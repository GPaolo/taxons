# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class AutoEncoder(nn.Module):

  def __init__(self, device=None, learning_rate=0.01, **kwargs):
    super(AutoEncoder, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.subsample = nn.AvgPool2d(8).cuda(self.device) # 600 -> 75

    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2), nn.LeakyReLU(), # 75 -> 36
                                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2), nn.LeakyReLU(), # 36 -> 17
                                nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1), nn.LeakyReLU()).to(self.device) # 17 -> 15

    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3, stride=1), nn.LeakyReLU(), # 8 -> 17
                                nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2), nn.LeakyReLU(), # 17 -> 36
                                nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=5, stride=2), nn.ReLU()).to(self.device) # 36 -> 75

    # self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2), nn.LeakyReLU(), # 75 -> 35
    #                              nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=3), nn.LeakyReLU()).to(self.device)  # 35 -> 11
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=5, stride=3), nn.LeakyReLU(),
    #                              nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=7, stride=2), nn.ReLU()).to(self.device)

    # self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, stride=2), nn.ReLU()).to(self.device)  # 75 -> 36
    #
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=5, stride=2), nn.ReLU()).to(self.device)

    self.criterion = nn.MSELoss()
    self.learning_rate = learning_rate
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)

    self.to(self.device)
    self.criterion.to(self.device)

  def _get_surprise(self, x):
    if x.shape[-1] > 75:  # Only subsample if not done yet.
      x = self.subsample(x)
    y, feat = self.forward(x)
    loss = self.criterion(x, y)
    return loss, feat

  def forward(self, x):
    if x.shape[-1] > 75:  # Only subsample if not done yet.
      x = self.subsample(x)
    feat = self.encoder(x)
    y = self.decoder(feat)
    return y, feat

  def training_step(self, x):
    self.optimizer.zero_grad()
    novelty, feat = self._get_surprise(x)
    novelty.backward()
    self.optimizer.step()
    return novelty, feat

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