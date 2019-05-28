# Created by Giuseppe Paolo
# Date: 20/02/19

import torch, torchvision
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

    self.first_subs = 256
    self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(self.first_subs),
                                   nn.AvgPool2d(2),
                                   nn.AvgPool2d(2)) # 256->64

    # Encoder
    # ------------------------------------
    self.encoder = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(), #64->32
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(), #32->16
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(), #16->8
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(),
                                 nn.BatchNorm2d(64),) #8->4
    self.encoder_ff = nn.Sequential(nn.Linear(64 * 4 * 4, 1024, bias=True), nn.LeakyReLU(),
                                    nn.Linear(1024, 256, bias=True), nn.LeakyReLU(),
                                    nn.Linear(256, kwargs['encoding_shape'], bias=True), nn.ReLU())
    # ------------------------------------

    # Decoder
    # ------------------------------------
    self.decoder_ff = nn.Sequential(nn.Linear(kwargs['encoding_shape'], 256, bias=True), nn.LeakyReLU(),
                                    nn.Linear(256, 32*4*4, bias=True), nn.LeakyReLU())
    self.decoder = nn.Sequential(nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(), # 4 -> 8
                                 nn.BatchNorm2d(64),
                                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(), # 8 -> 16
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(), # 16 -> 32
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True), nn.ReLU()) # 32 -> 64
    # ------------------------------------

    self.feat_reg = nn.L1Loss()
    self.rec_loss = nn.MSELoss(reduction='none')
    self.learning_rate = learning_rate
    self.zero_grad()

    self.ae_optimizer = optim.Adam(self.parameters(), self.learning_rate)

    self.to(self.device) 
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    if x.shape[-1] > self.first_subs/4:  # Only subsample if not done yet.
      x = self.subsample(x)

    feat = self.encoder(x)
    feat = feat.view(-1, 64*4*4)
    feat = self.encoder_ff(feat)
    y = self.decoder_ff(feat)
    y = y.view(-1, 32, 4,4)
    y = self.decoder(y)

    rec_error = self.rec_loss(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims) # Reconstruction error for each sample

    return rec_error, torch.squeeze(feat), y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x, old_x=None):
    if x.shape[-1] > self.first_subs/4:  # Only subsample if not done yet.
      x = self.subsample(x)

    rec_error, feat, y = self.forward(x)
    # Features Regularization
    zeros = torch.zeros(feat.shape).to(self.device)
    feat_reg = self.feat_reg(feat, zeros)
    # Reconstruction Loss
    rec_loss = torch.mean(rec_error)
    # Maximum entropy prior on features
    mid = int(len(feat)/2)
    variability = torch.mean(torch.exp(-torch.abs(feat[:mid] - feat[mid:2*mid])))


    loss = rec_loss + feat_reg*0.01 + variability
    loss.backward()

    self.ae_optimizer.step()
    return loss, feat, y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def save(self, filepath):
    save_ckpt = {
      'ae': self.state_dict(),
      'optimizer': self.ae_optimizer.state_dict()
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
      self.ae_optimizer.load_state_dict(ckpt['optimizer'])
    except Exception as e:
      print('Could not load optimizer state dict: {}'.format(e))
  # ----------------------------------------------------------------
