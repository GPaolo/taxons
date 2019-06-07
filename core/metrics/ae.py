# Created by Giuseppe Paolo
# Date: 20/02/19

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import os
import sys
from core.utils import utils

# ----------------------------------------------------------------
class View(nn.Module):
  def __init__(self, size):
    super(View, self).__init__()
    self.size = size

  def forward(self, tensor):
    return tensor.view(self.size)
# ----------------------------------------------------------------

class BaseAE(nn.Module):
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
      ckpt = torch.load(filepath, map_location=self.device)
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

  # ----------------------------------------------------------------
  def init_layers(self, m):
    classname = m.__class__.__name__
    if 'Conv' in classname and not 'Encoder' in classname:
      m.weight.data.normal_(0.0, 1/(m.kernel_size[0]*m.kernel_size[1]))
    elif 'Linear' in classname:
      m.weight.data.normal_(0.0, 1/m.in_features)
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, **kwargs):
    raise NotImplementedError
  # ----------------------------------------------------------------


class AutoEncoder(BaseAE):
  # ----------------------------------------------------------------
  def __init__(self, device=None, learning_rate=0.001, lr_scale=None, **kwargs):
    super(AutoEncoder, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.encoding_shape = kwargs['encoding_shape']
    self.first_subs = 256
    self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(self.first_subs),
                                   nn.AvgPool2d(2),
                                   nn.AvgPool2d(2)) # 256->64

    # Encoder
    # ------------------------------------
    self.encoder = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(), #64->32
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(), #32->16
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(), #16->8
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(), #8->4
                                 nn.BatchNorm2d(64),
                                 View((-1, 64 * 4 * 4)),
                                 nn.Linear(64 * 4 * 4, 1024, bias=False), nn.SELU(),
                                 nn.Linear(1024, 256, bias=False), nn.SELU(),
                                 nn.Linear(256, self.encoding_shape, bias=False), nn.SELU(),
                                 )
    # ------------------------------------

    # Decoder
    # ------------------------------------
    self.decoder = nn.Sequential(nn.Linear(self.encoding_shape, 256, bias=False), nn.SELU(),
                                 nn.Linear(256, 32 * 4 * 4, bias=False), nn.SELU(),
                                 View((-1, 32, 4, 4)),
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(), # 4 -> 8
                                 nn.BatchNorm2d(64),
                                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(), # 8 -> 16
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(), # 16 -> 32
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(), # 32 -> 64
                                 )
    # ------------------------------------

    self.rec_loss = nn.MSELoss(reduction='none')
    self.learning_rate = learning_rate
    self.zero_grad()

    self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
    self.lr_scale = lr_scale
    if self.lr_scale is not None:
      self.lr_scheduler = utils.LRScheduler(self.optimizer, self.lr_scale)
    self.to(self.device)
    self.eval()
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    if x.shape[-1] > self.first_subs/4:  # Only subsample if not done yet.
      x = self.subsample(x)

    feat = self.encoder(x)
    y = self.decoder(feat)

    rec_error = self.rec_loss(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims) # Reconstruction error for each sample

    return rec_error, torch.squeeze(feat), y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x, old_x=None):
    self.train()
    rec_error, feat, y = self.forward(x)
    # Reconstruction Loss
    rec_loss = torch.mean(rec_error)
    loss = rec_loss

    self.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.eval()
    print('Rec Loss: {}'.format(rec_loss.cpu().data))
    print()
    return loss, feat, y
  # ----------------------------------------------------------------


class BVAE(BaseAE):
  # ----------------------------------------------------------------
  def __init__(self, device=None, learning_rate=0.001, lr_scale=None, **kwargs):
    """
    Beta-VAE implementation taken from https://github.com/1Konny/Beta-VAE/blob/master/model.py
    :param device:
    :param learning_rate:
    :param kwargs:
    """
    super(BVAE, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.encoding_shape = kwargs['encoding_shape']
    self.first_subs = 256
    self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(self.first_subs),
                                   nn.AvgPool2d(2),
                                   nn.AvgPool2d(2))  # 256->64
    # Encoder
    # ------------------------------------
    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),  # B,  32, 32, 32
                                 nn.SELU(),
                                 nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                                 nn.SELU(),
                                 nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                                 nn.SELU(),
                                 nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                                 nn.SELU(),
                                 nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                                 nn.SELU(),
                                 View((-1, 256 * 1 * 1)),  # B, 256
                                 nn.Linear(256, self.encoding_shape * 2),  # B, z_dim*2
                                 )
    # ------------------------------------

    # Decoder
    # ------------------------------------
    self.decoder = nn.Sequential(nn.Linear(self.encoding_shape, 256),  # B, 256
                                 View((-1, 256, 1, 1)),  # B, 256,  1,  1
                                 nn.SELU(),
                                 nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
                                 nn.SELU(),
                                 nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
                                 nn.SELU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
                                 nn.SELU(),
                                 nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
                                 nn.SELU(),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1),  # B, nc, 64, 64
                                 nn.ReLU()
                                 )
    # ------------------------------------

    self.rec_loss = nn.MSELoss(reduction='none')
    self.learning_rate = learning_rate
    self.beta = kwargs['beta']
    self.zero_grad()

    self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
    self.lr_scale = lr_scale
    if self.lr_scale is not None:
      self.lr_scheduler = utils.LRScheduler(self.optimizer, self.lr_scale)
    self.to(self.device)
    self.eval()
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def reparametrize(self, mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.Tensor(std.data.new(std.size()).normal_())
    return mu + std * eps
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    if x.shape[-1] > self.first_subs/4:  # Only subsample if not done yet.
      x = self.subsample(x)

    distributions = self.encoder(x)
    mu = distributions[:, :self.encoding_shape]
    logvar = distributions[:, self.encoding_shape:]
    feat = self.reparametrize(mu, logvar)
    y = self.decoder(feat)

    rec_error = self.rec_loss(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims)  # Reconstruction error for each sample

    if self.training:
      return rec_error, torch.squeeze(feat), y, mu, logvar
    else:
      return rec_error, torch.squeeze(feat), y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def kl_divergence(self, mu, logvar):
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x, old_x=None):
    self.train()
    rec_error, feat, y, mu, logvar = self.forward(x)
    # Reconstruction Loss
    rec_loss = torch.mean(rec_error)
    # KL divergence
    total_kld, dim_kld, mean_kld = self.kl_divergence(mu, logvar)

    # Final loss
    loss = rec_loss + self.beta * total_kld

    self.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.eval()
    print('Rec Loss: {}'.format(rec_loss.cpu().data))
    print('Total Loss: {}'.format(loss.cpu().data))
    print()
    return loss, feat, y
  # ----------------------------------------------------------------


