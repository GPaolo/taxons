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
  """
  This class is used to reshape tensors
  """
  def __init__(self, size):
    """
    Constructor
    :param size: final shape of the tensor
    """
    super(View, self).__init__()
    self.size = size

  def forward(self, tensor):
    return tensor.view(self.size)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
class BaseAE(nn.Module):
  """
  This class defines the Base AutoEncoder.
  """
  # ----------------------------------------------------------------
  def __init__(self, device=None, learning_rate=0.001, lr_scale=None, **kwargs):
    """
    Constructor
    :param device: Device on which run the computation
    :param learning_rate:
    :param lr_scale: Learning rate scale for the lr scheduler
    :param kwargs:
    """
    super(BaseAE, self).__init__()

    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.encoding_shape = kwargs['encoding_shape']
    # Model definition is done in these functions that are to be overridden
    self._define_subsampler()
    self._define_encoder()
    self._define_decoder()

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
  def _define_encoder(self):
    """
    Define encoder. Needs to be implemented in inheriting classes
    """
    raise NotImplementedError

  def _define_decoder(self):
    """
    Define decoder. Needs to be implemented in inheriting classes
    """
    raise NotImplementedError

  def _define_subsampler(self):
    """
    Define subsampler.
    """
    self.first_subs = 256
    self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(self.first_subs),
                                   nn.AvgPool2d(2),
                                   nn.AvgPool2d(2))  # 256->64
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def save(self, filepath):
    """
    Saves AE to given filepath
    :param filepath:
    """
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
    """
    Load AE from given filepath
    :param filepath:
    """
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
  def training_step(self, **kwargs):
    """
    Function that performs one training step. Needs to be implemented in inheriting classes
    :param kwargs:
    """
    raise NotImplementedError
  # ----------------------------------------------------------------
# ----------------------------------------------------------------

# ----------------------------------------------------------------
class ConvAE(BaseAE):
  """
  This class implements the Convolutional Autoencoder
  """
  # ----------------------------------------------------------------
  def _define_encoder(self):
    """
    Defines encoder
    """
    self.encoder = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),  # 64->32
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),
                                 # 32->16
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),
                                 # 16->8
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),  # 8->4
                                 nn.BatchNorm2d(64),
                                 View((-1, 64 * 4 * 4)),
                                 nn.Linear(64 * 4 * 4, 1024, bias=False), nn.SELU(),
                                 nn.Linear(1024, 256, bias=False), nn.SELU(),
                                 nn.Linear(256, self.encoding_shape, bias=False), nn.SELU(),
                                 )
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _define_decoder(self):
    """
    Defines the decoder
    """
    self.decoder = nn.Sequential(nn.Linear(self.encoding_shape, 256, bias=False), nn.SELU(),
                                 nn.Linear(256, 32 * 4 * 4, bias=False), nn.SELU(),
                                 View((-1, 32, 4, 4)),
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),
                                 # 4 -> 8
                                 nn.BatchNorm2d(64),
                                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),
                                 # 8 -> 16
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),
                                 # 16 -> 32
                                 nn.BatchNorm2d(32),
                                 nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(),
                                 # 32 -> 64
                                 )
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    """
    Forward pass of the network.
    :param x: Input as RGB array of images
    :return: reconstruction error, features, reconstructed image
    """
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
  def training_step(self, x):
    """
    Performs one training step of the network
    :param x: Input
    :return: Loss, features, reconstructed image
    """
    self.train() # Sets network to train mode
    rec_error, feat, y = self.forward(x)
    # Reconstruction Loss
    rec_loss = torch.mean(rec_error)
    loss = rec_loss

    self.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.eval() # Sets network to evaluation mode
    print('Rec Loss: {}'.format(rec_loss.cpu().data))
    print()
    return loss, feat, y
  # ----------------------------------------------------------------
# ----------------------------------------------------------------

# ----------------------------------------------------------------
class FFAE(BaseAE):
  """
  This class implements a FF autoencoder
  """
  # ----------------------------------------------------------------
  def _define_encoder(self):
    """
    Defines encoder of the network
    """
    self.encoder = nn.Sequential(View((-1, 64 * 64 * 3)),
                                 nn.Linear(64 * 64 * 3, 5120, bias=False), nn.SELU(),
                                 nn.BatchNorm1d(5120),
                                 nn.Linear(5120, 2560, bias=False), nn.SELU(),
                                 nn.BatchNorm1d(2560),
                                 nn.Linear(2560, 512, bias=False), nn.SELU(),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, 128, bias=False), nn.SELU(),
                                 nn.BatchNorm1d(128),
                                 nn.Linear(128, self.encoding_shape, bias=False), nn.SELU(),
                                 )
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _define_decoder(self):
    """
    Defines decoder of the network
    """
    self.decoder = nn.Sequential(nn.Linear(self.encoding_shape, 512, bias=False), nn.SELU(),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, 2560, bias=False), nn.SELU(),
                                 nn.BatchNorm1d(2560),
                                 nn.Linear(2560, 5120, bias=False), nn.SELU(),
                                 nn.BatchNorm1d(5120),
                                 nn.Linear(5120, 64*64*3, bias=False), nn.ReLU(),
                                 View((-1, 3, 64, 64)),
                                 )
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    """
    Performs forward pass of the net
    :param x: Input
    :return: rec_error, features, reconstructed image
    """
    if x.shape[-1] > self.first_subs/4:  # Only subsample if not done yet.
      x = self.subsample(x)

    feat = self.encoder(x)
    y = self.decoder(feat)

    rec_error = self.rec_loss(x, y)
    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(rec_error.shape)))
    rec_error = torch.mean(rec_error, dim=dims)  # Reconstruction error for each sample

    return rec_error, torch.squeeze(feat), y
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x):
    """
    Performs one training step of the network
    :param x: Input
    :return: Loss, features, reconstructed image
    """
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
# ----------------------------------------------------------------

# ----------------------------------------------------------------
class BVAE(BaseAE):
  """
  Beta-VAE implementation taken from https://github.com/1Konny/Beta-VAE/blob/master/model.py
  """
  # ----------------------------------------------------------------
  def __init__(self, device=None, learning_rate=0.001, lr_scale=None, **kwargs):
    """
    Constructor
    :param device: Device on which run the computation
    :param learning_rate:
    :param lr_scale: Learning rate scale for the lr scheduler
    :param kwargs:
    """
    super(BVAE, self).__init__(device, learning_rate, lr_scale, **kwargs)

    self.beta = kwargs['beta']
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _define_encoder(self):
    """
    Defines encoder of the net
    """
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
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _define_decoder(self):
    """
    Define decoder of the net
    """
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
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def reparametrize(self, mu, logvar):
    """
    Reparametrize the mu and logvar to obtain the features
    :param mu:
    :param logvar:
    :return: mu + std * eps
    """
    std = logvar.div(2).exp()
    eps = torch.Tensor(std.data.new(std.size()).normal_())
    return mu + std * eps
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    """
    Forward pass of the network
    :param x: Input
    :return: If training: Rec_error, features, reconstructed image, mu, logvar
             If not training: rec error, features, reconstructed image
    """
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
    """
    Calculate KL divergence
    :param mu:
    :param logvar:
    :return: total kld, dimension wise kld, mean kld
    """
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x):
    """
    Performs training step
    :param x: Input
    :return: loss, features, reconstructed image
    """
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
# ----------------------------------------------------------------


