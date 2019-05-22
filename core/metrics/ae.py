# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import operator
import functools
import pytorch_msssim
from pytorch_msssim import msssim, ssim


class ConvAutoEncoder(nn.Module):

  # ----------------------------------------------------------------
  def __init__(self, device=None, learning_rate=0.001, **kwargs):
    super(ConvAutoEncoder, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(512),
                                   nn.AvgPool2d(2),
                                   nn.AvgPool2d(2)) # 512 -> 128

    # Encoder
    # ------------------------------------
    self.encoder = nn.Sequential(nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(), #64->32
                                 nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(), #32->16
                                 nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(), #16->8
                                 nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU()) #8->4
    self.encoder_ff = nn.Sequential(nn.Linear(16 * 8 * 8, 512, bias=False), nn.LeakyReLU(),
                                    nn.Linear(512, 256, bias=False), nn.LeakyReLU(),
                                    nn.Linear(256, kwargs['encoding_shape'], bias=False))
    # ------------------------------------

    # Decoder
    # ------------------------------------
    self.decoder_ff = nn.Sequential(nn.Linear(kwargs['encoding_shape'], 256, bias=False), nn.LeakyReLU(),
                                    nn.Linear(256, 16*8*8, bias=False), nn.LeakyReLU())
    self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(), # 4 -> 8
                                 nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(), # 8 -> 16
                                 nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(), # 16 -> 32
                                 nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, bias=False), nn.Sigmoid()) # 32 -> 64
    # ------------------------------------

    #self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, bias=False), nn.LeakyReLU(),  # 75 -> 35
    #                              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=3, bias=False), nn.LeakyReLU(),  # 35 -> 11
    #                              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(), # 11 -> 7
    #                              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(), # 7 -> 3
    #                              nn.Conv2d(in_channels=32, out_channels=kwargs['encoding_shape'], kernel_size=3, bias=False), nn.LeakyReLU()).to(self.device)  # 3 -> 1
    #
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=kwargs['encoding_shape'], out_channels=32, kernel_size=3, bias=False), nn.LeakyReLU(),  # 1 -> 3
    #                              nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(),  # 3 -> 7
    #                              nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, bias=False), nn.LeakyReLU(),  # 7 -> 11
    #                              nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, bias=False), nn.LeakyReLU(), # 11 -> 35
    #                              nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=7, stride=2, bias=False), nn.LeakyReLU()).to(self.device)  # 35 -> 75
    #self.encoder_ff = nn.Sequential(nn.Linear(484, kwargs['encoding_shape']), nn.LeakyReLU()).to(self.device)
    #self.decoder_ff = nn.Sequential(nn.Linear(kwargs['encoding_shape'], 484, bias=False), nn.LeakyReLU()).to(self.device)

    self.loss = pytorch_msssim.SSIM()
    self.learning_rate = learning_rate
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

    self.to(self.device)
    self.loss.to(self.device)
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    if x.shape[-1] > 128:  # Only subsample if not done yet.
      x = self.subsample(x)
    feat, y = self._get_reconstruction(x)

    rec_error = (x-y)**2
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims) # Reconstruction error for each sample

    return rec_error, feat, y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _get_reconstruction(self, x):
    if x.shape[-1] > 128:  # Only subsample if not done yet.
      x = self.subsample(x)

    feat = self.encoder(x)
    shape = feat.shape
    feat = feat.view(-1, 16*8*8)
    feat = self.encoder_ff(feat)

    y = self.decoder_ff(feat)
    y = y.view(shape)
    y = self.decoder(y)

    feat = torch.squeeze(feat)
    return feat, y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x):
    if x.shape[-1] > 128:  # Only subsample if not done yet.
      x = self.subsample(x)

    self.optimizer.zero_grad()
    _, feat, y = self.forward(x)

    loss = 1 - self.loss(x, y)
    loss.backward()

    self.optimizer.step()
    return loss, feat, y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def save(self, filepath):
    save_ckpt = {
      'ae': self.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    try:
      torch.save(save_ckpt, os.path.join(filepath, 'ckpt_ae.pth'))
    except:
      print('Cannot save autoencoder.')
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
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
  # ----------------------------------------------------------------
