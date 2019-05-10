# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys


class ConvAutoEncoder(nn.Module):

  def __init__(self, device=None, learning_rate=0.001, **kwargs):
    super(ConvAutoEncoder, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(300),
                                   nn.AvgPool2d(2),
                                   nn.AvgPool2d(2)).to(self.device) # 600 -> 75

    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, bias=False), nn.LeakyReLU(),  # 75 -> 35
                                 nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=3, bias=False), nn.LeakyReLU(),  # 35 -> 11
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(), # 11 -> 7
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(), # 7 -> 3
                                 nn.Conv2d(in_channels=32, out_channels=kwargs['encoding_shape'], kernel_size=3, bias=False), nn.LeakyReLU()).to(self.device)  # 3 -> 1

    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=kwargs['encoding_shape'], out_channels=32, kernel_size=3, bias=False), nn.LeakyReLU(),  # 1 -> 3
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(),  # 3 -> 7
                                 nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, bias=False), nn.LeakyReLU(),  # 7 -> 11
                                 nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, bias=False), nn.LeakyReLU(), # 11 -> 35
                                 nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=7, stride=2, bias=False), nn.LeakyReLU()).to(self.device)  # 35 -> 75

    self.criterion = nn.MSELoss(reduction='none')
    self.learning_rate = learning_rate
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)

    self.to(self.device)
    self.criterion.to(self.device)

  def forward(self, x):
    if x.shape[-1] > 75:  # Only subsample if not done yet.
      x = self.subsample(x)
    feat, y = self._get_reconstruction(x)

    rec_error = self.criterion(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims)

    return rec_error, feat, y

  def _get_reconstruction(self, x):
    if x.shape[-1] > 75:  # Only subsample if not done yet.
      x = self.subsample(x)
    feat = self.encoder(x)
    y = self.decoder(feat)

    feat = torch.squeeze(feat)
    return feat, y

  def training_step(self, x):
    self.optimizer.zero_grad()
    novelty, feat, y = self.forward(x)
    novelty = torch.mean(novelty)
    novelty.backward()
    self.optimizer.step()
    return novelty, feat, y

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


class FFAutoEncoder(nn.Module):

  def __init__(self, device=None, learning_rate=0.01, **kwargs):
    super(FFAutoEncoder, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.encoder = nn.Sequential(nn.Linear(14, 10), nn.Tanh(), # 14 -> 10
                                 nn.Linear(10, kwargs['encoding_shape']), nn.Tanh()).to(self.device)  # 10 -> 5

    self.decoder = nn.Sequential(nn.Linear(kwargs['encoding_shape'], 10), nn.Tanh(),
                                 nn.Linear(10, 14), nn.Tanh()).to(self.device)


    self.criterion = nn.MSELoss(reduction='none')
    self.learning_rate = learning_rate
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)

    self.to(self.device)
    self.criterion.to(self.device)

  def _get_surprise(self, x):
    y, feat = self.forward(x)
    loss = self.criterion(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(loss.shape)))
    loss = torch.mean(loss, dim=dims)

    return loss, feat

  def forward(self, x):
    feat = self.encoder(x)
    y = self.decoder(feat)
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