# Created by Giuseppe Paolo
# Date: 20/02/19

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Link(object):
  def __init__(self, in_node, out_node, value):
    """
    Creates new network edge connecting particular network nodes
    :param: node_in: the ID of inputing node
    :param: node_out: the ID of node this link affects
    :param: value: the weight of this link
    """
    self.in_node = in_node
    self.out_node = out_node
    self.value = value


class Node(object):
  def __init__(self, id, type, activation):
    self.id = id
    self.type = type
    self.activation = activation

class NEATAgent(object):

  def __init__(self, input_size, output_size):
    self.input_size = input_size
    self.output_size = output_size

    self.total_nodes = self.input_size + self.output_size
    self.connection_matrix = np.random.rand(self.total_nodes-self.output_size, self.total_nodes-self.input_size) > 0.5
    self.weights = np.random.random(self.connection_matrix.shape)
    self.matrix_shape = np.array(self.connection_matrix.shape)

    self.add_node_prob = 0.05
    self.add_conn_prob = 0.05
    self.mod_weight_prob = 0.05

  def _add_node(self):
    # if np.random.uniform() <= self.add_node_prob:
    self.matrix_shape += 1
    h = np.random.rand(self.matrix_shape[0]-1, 1) <= self.add_conn_prob
    v = np.random.rand(1, self.matrix_shape[1]) <= self.add_conn_prob


    self.connection_matrix = np.hstack((self.connection_matrix, h))
    self.connection_matrix = np.vstack((self.connection_matrix, v))
    self.weights = np.hstack((self.weights, np.zeros_like(h)))
    self.weights = np.vstack((self.weights, np.zeros_like(v)))


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

    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, bias=False), nn.LeakyReLU(), # 75 -> 35
                                 nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=3, bias=False), nn.LeakyReLU()).to(self.device)  # 35 -> 11

    self.encoder_ff = nn.Sequential(nn.Linear(484, 128), nn.LeakyReLU(), nn.Linear(128, kwargs['encoding_shape'], bias=False), nn.LeakyReLU()).to(self.device)
    self.decoder_ff = nn.Sequential(nn.Linear(kwargs['encoding_shape'], 484, bias=False), nn.LeakyReLU()).to(self.device)


    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=5, stride=3, bias=False), nn.LeakyReLU(),
                                 nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=7, stride=2, bias=False), nn.ReLU()).to(self.device)


    self.criterion = nn.MSELoss(reduction='none')
    self.learning_rate = learning_rate
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

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


if __name__ == "__main__":
  with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
    images = np.load(f)

  ae = ConvAutoEncoder(encoding_shape=32, device=torch.device("cuda"))
  images = torch.Tensor(images).permute(0, 3, 1, 2).to(torch.device("cuda"))/255.
  for k in range(5000):
    loss, _ = ae.training_step(images)
    print(loss)

  uu = 5
  y, a = ae.forward(images[uu:uu+1])
  print()
  print(a)


  import matplotlib.pyplot as plt
  y = y.permute(0, 2, 3, 1)[0]
  plt.figure()

  img = np.array(y.cpu().data*255)
  img = img.astype('int32')
  plt.imshow(img)
  plt.show()