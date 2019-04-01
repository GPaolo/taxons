# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys


class AutoEncoder(nn.Module):

  def __init__(self, device=None, learning_rate=0.01, **kwargs):
    super(AutoEncoder, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.subsample = nn.MaxPool2d(8).to(self.device) # 600 -> 75

    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2), nn.LeakyReLU(), # 75 -> 35
                                 nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=3), nn.LeakyReLU()).to(self.device)  # 35 -> 11

    self.encoder_ff = nn.Sequential(nn.Linear(484, kwargs['encoding_shape']), nn.LeakyReLU()).to(self.device)
    self.decoder_ff = nn.Sequential(nn.Linear(kwargs['encoding_shape'], 484), nn.LeakyReLU()).to(self.device)


    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=5, stride=3), nn.LeakyReLU(),
                                 nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=7, stride=2), nn.ReLU()).to(self.device)


    self.criterion = nn.MSELoss(reduction='none')
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
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(loss.shape)))
    loss = torch.mean(loss, dim=dims)

    return loss, feat

  def forward(self, x):
    if x.shape[-1] > 75:  # Only subsample if not done yet.
      x = self.subsample(x)
    feat = self.encoder(x)

    shape = feat.shape
    feat = feat.view(-1, 484)

    feat = self.encoder_ff(feat)
    y = self.decoder_ff(feat)
    y = y.view(shape)

    y = self.decoder(y)
    return y, feat

  def training_step(self, x):
    self.optimizer.zero_grad()
    novelty, feat = self._get_surprise(x)
    novelty = torch.mean(novelty)
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

  def load(self, filepath):
    try:
      ckpt = torch.load(filepath)
    except Exception as e:
      print('Could not load file: {}'.format(e))
      sys.exit()
    try:
      self.load_state_dict(ckpt['ae'])
    except Exception as e:
      print('Could not load model state dict: {}'.format(e))
    try:
      self.optimizer.load_state_dict(ckpt['optimizer'])
    except Exception as e:
      print('Could not load optimizer state dict: {}'.format(e))
