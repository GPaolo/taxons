# Created by Giuseppe Paolo
# Date: 20/02/19
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
from core.utils import utils

# ----------------------------------------------------------------
class View(nn.Module):
  def __init__(self, size):
    super(View, self).__init__()
    self.size = size

  def forward(self, tensor):
    return tensor.view(self.size)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
class BaseNet(nn.Module):
  '''
    This class defines the networks used for the RND
    '''
  # ----------------------------------------------------------------
  def __init__(self, encoding_shape, fixed=True):
    super(BaseNet, self).__init__()

    self.encoding_shape = encoding_shape
    self.fixed = fixed

    if fixed: # That is the target
      self.model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=True), nn.SELU(),  # 64->32
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1, bias=True), nn.SELU(),
                                 # 32->16
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True), nn.SELU(),
                                 # 16->8
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.SELU(),  # 8->4
                                 nn.BatchNorm2d(64),
                                 View((-1, 64 * 4 * 4)),
                                 nn.Linear(64 * 4 * 4, 1024, bias=True), nn.SELU(),
                                 nn.Linear(1024, 256, bias=True), nn.SELU(),
                                 nn.Linear(256, self.encoding_shape, bias=True), nn.SELU(),
                                 )
    else: # That is the predictor
      self.model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),  # 64->32
                                 nn.BatchNorm2d(16),
                                 nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),
                                 # 32->16
                                 nn.BatchNorm2d(64),
                                 nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),
                                 # 16->8
                                 nn.BatchNorm2d(64),
                                 nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.SELU(),  # 8->4
                                 nn.BatchNorm2d(32),
                                 View((-1, 32 * 4 * 4)),
                                 nn.Linear(32 * 4 * 4, 512, bias=False), nn.SELU(),
                                 nn.Linear(512, self.encoding_shape, bias=False), nn.SELU(),
                                 )
    for l in self.parameters():
      l.requires_grad = not self.fixed
    self.model.apply(self.init_layers)
    self.zero_grad()
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def init_layers(self, m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
      m.weight.data.normal_(0.0, 1)
    elif 'Linear' in classname:
      m.weight.data.normal_(0.0, 1 / m.in_features)
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    feat = self.model(x)
    return feat
  # ----------------------------------------------------------------
# ----------------------------------------------------------------

# ----------------------------------------------------------------
class RND(nn.Module):

  # ----------------------------------------------------------------
  def __init__(self, encoding_shape, learning_rate=0.0001, lr_scale=None, device=None):
    '''
    Class that instantiates the RND component
    '''
    super(RND, self).__init__()
    self.encoding_shape = encoding_shape
    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")


    self.first_subs = 256
    self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(self.first_subs),
                                   nn.AvgPool2d(2),
                                   nn.AvgPool2d(2))  # 256->64
    # Nets
    self.target_model = BaseNet(encoding_shape=self.encoding_shape, fixed=True)
    self.predictor_model = BaseNet(encoding_shape=self.encoding_shape, fixed=False)

    # Loss
    self.criterion = nn.MSELoss(reduction='none')

    # Optimizer
    self.learning_rate = learning_rate
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
    self.lr_scale = lr_scale
    if self.lr_scale is not None:
      self.lr_scheduler = utils.LRScheduler(self.optimizer, self.lr_scale)
    self.to(self.device)
    self.eval()
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def forward(self, x):
    '''
    This function calculates the surprise given by the input
    :param x: Network input. Needs to be a torch tensor.
    :return: surprise as a 1 dimensional torch tensor
    '''
    if x.shape[-1] > self.first_subs/4:  # Only subsample if not done yet.
      x = self.subsample(x)

    target = self.target_model(x)
    prediction = self.predictor_model(x)
    loss = self.criterion(prediction, target)

    # Make mean along all the dimensions except the batch one
    dims = list(range(1, len(loss.shape)))
    loss = torch.mean(loss, dim=dims)  # Reconstruction error for each sample

    return loss, torch.squeeze(prediction), None
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, x):
    '''
    This function performs the training step.
    :param x: Network input. Needs to be a torch tensor.
    :return: surprise as a 1 dimensional torch tensor
    '''
    self.train()
    self.zero_grad()
    surprise, feat, _ = self.forward(x)
    surprise = torch.mean(surprise)
    surprise.backward()

    self.optimizer.step()
    self.eval()
    print('Rec Loss: {}'.format(surprise.cpu().data))
    print()
    return surprise, feat, None
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def save(self, filepath):
    save_ckpt = {
      'target_model': self.target_model.state_dict(),
      'predictor_model': self.predictor_model.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    try:
      torch.save(save_ckpt, os.path.join(filepath, 'ckpt_rnd.pth'))
    except:
      print('Cannot save rnd networks.')
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def load(self, filepath):
    try:
      ckpt = torch.load(filepath, map_location=self.device)
    except Exception as e:
      print('Could not load file: {}'.format(e))
      sys.exit()
    try:
      self.target_model.load_state_dict(ckpt['target_model'])
    except Exception as e:
      print('Could not load target model state dict: {}'.format(e))
    try:
      self.predictor_model.load_state_dict(ckpt['predictor_model'])
    except Exception as e:
      print('Could not load predictor model state dict: {}'.format(e))
    try:
      self.optimizer.load_state_dict(ckpt['optimizer'])
    except Exception as e:
      print('Could not load optimizer state dict: {}'.format(e))
  # ----------------------------------------------------------------



if __name__ == '__main__':
  device = torch.device('cpu')
  rnd = RND(6, 2, device=device)

  import numpy as np
  state = np.load('../../env.npy')/255
  state = torch.Tensor(state).permute(2,0,1).unsqueeze(0)

  for i in range(1000):
    aaa = rnd.training_step(state)
    print(aaa)

