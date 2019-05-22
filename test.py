# # Created by Giuseppe Paolo
# # Date: 20/02/19
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# #
# # class Link(object):
# #   def __init__(self, in_node, out_node, value):
# #     """
# #     Creates new network edge connecting particular network nodes
# #     :param: node_in: the ID of inputing node
# #     :param: node_out: the ID of node this link affects
# #     :param: value: the weight of this link
# #     """
# #     self.in_node = in_node
# #     self.out_node = out_node
# #     self.value = value
# #
# #
# # class Node(object):
# #   def __init__(self, id, type, activation):
# #     self.id = id
# #     self.type = type
# #     self.activation = activation
# #
# # class NEATAgent(object):
# #
# #   def __init__(self, input_size, output_size):
# #     self.input_size = input_size
# #     self.output_size = output_size
# #
# #     self.total_nodes = self.input_size + self.output_size
# #     self.connection_matrix = np.random.rand(self.total_nodes-self.output_size, self.total_nodes-self.input_size) > 0.5
# #     self.weights = np.random.random(self.connection_matrix.shape)
# #     self.matrix_shape = np.array(self.connection_matrix.shape)
# #
# #     self.add_node_prob = 0.05
# #     self.add_conn_prob = 0.05
# #     self.mod_weight_prob = 0.05
# #
# #   def _add_node(self):
# #     # if np.random.uniform() <= self.add_node_prob:
# #     self.matrix_shape += 1
# #     h = np.random.rand(self.matrix_shape[0]-1, 1) <= self.add_conn_prob
# #     v = np.random.rand(1, self.matrix_shape[1]) <= self.add_conn_prob
# #
# #
# #     self.connection_matrix = np.hstack((self.connection_matrix, h))
# #     self.connection_matrix = np.vstack((self.connection_matrix, v))
# #     self.weights = np.hstack((self.weights, np.zeros_like(h)))
# #     self.weights = np.vstack((self.weights, np.zeros_like(v)))
#
#
# class ConvAutoEncoder(nn.Module):
#
#   def __init__(self, device=None, learning_rate=0.001, **kwargs):
#     super(ConvAutoEncoder, self).__init__()
#
#     if device is not None:
#       self.device = device
#     else:
#       self.device = torch.device("cpu")
#
#     self.subsample = nn.Sequential(nn.AdaptiveAvgPool2d(300),
#                                    nn.AvgPool2d(2),
#                                    nn.AvgPool2d(2)).to(self.device) # 600 -> 75
#
#     self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, bias=False), nn.LeakyReLU(), # 75 -> 35
#                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=3, bias=False), nn.LeakyReLU(), # 35 -> 11
#                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(), # 11 -> 7
#                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(), # 7 -> 3
#                                  nn.Conv2d(in_channels=32, out_channels=kwargs['encoding_shape'], kernel_size=3, bias=False), nn.LeakyReLU()).to(self.device)  # 3 -> 1
#
#     self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=kwargs['encoding_shape'], out_channels=32, kernel_size=3, bias=False), nn.LeakyReLU(), # 1 -> 3
#                                  nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, bias=False), nn.LeakyReLU(),  # 3 -> 7
#                                  nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, bias=False), nn.LeakyReLU(),  # 7 -> 11
#                                  nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, bias=False), nn.LeakyReLU(),  # 11 -> 35
#                                  nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=7, stride=2, bias=False), nn.LeakyReLU()).to(self.device)  # 35 -> 75
#
#     self.criterion = nn.MSELoss(reduction='none')
#     self.learning_rate = learning_rate
#     self.zero_grad()
#     self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
#     self.att = nn.Sequential(nn.Linear(kwargs['encoding_shape'], kwargs['encoding_shape'], bias=False), nn.Softmax())
#
#     self.to(self.device)
#     self.criterion.to(self.device)
#
#   def forward(self, x, rew):
#     if x.shape[-1] > 75:  # Only subsample if not done yet.
#       x = self.subsample(x)
#     y, feat = self._get_reconstruction(x, rew)
#     rec_error = self.criterion(x, y)
#     # Make mean along all the dimensions except the batch one
#     dims = list(range(1, len(rec_error.shape)))
#     rec_error = torch.mean(rec_error, dim=dims)
#
#     return rec_error, feat, y
#
#   def _get_reconstruction(self, x, r):
#     if x.shape[-1] > 75:  # Only subsample if not done yet.
#       x = self.subsample(x)
#     feat = self.encoder(x)
#
#     # shape = feat.shape
#     # feat = feat.view(-1, 484)
#
#     # feat = self.encoder_ff(feat)
#     # y = self.decoder_ff(feat)
#     # y = y.view(shape)
#
#     y = self.decoder(feat)
#     feat = torch.squeeze(feat)
#     return y, feat
#
#   def training_step(self, x, rew):
#     self.optimizer.zero_grad()
#     rec_error, feat, y = self.forward(x, rew)
#     rec_error = torch.mean(rec_error)
#     novelty = rec_error
#
#     # att_feat = self.att(feat)
#     # print("ATT {}".format(att_feat))
#
#     loss = rec_error# + torch.mean(torch.sum(att_feat*feat*rew, dim=1), dim=0)
#     # print("Sparseness: {}".format(sparseness))
#
#     loss.backward()
#     self.optimizer.step()
#     return novelty, feat
#
#   def __call__(self, x):
#     return self._get_surprise(x)
#
#
# if __name__ == "__main__":
#   with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
#     images = np.load(f)
#
#   r = np.zeros((23, 1))
#   r[0] = 1.
#   r[5] = 0.7
#   r[3] = -0.5
#   r[9] = -0.9
#   r[7] = 0.2
#   r[17] = 1.
#
#   ae = ConvAutoEncoder(encoding_shape=8, device=torch.device("cuda"))
#   images = torch.Tensor(images).permute(0, 3, 1, 2).to(torch.device("cuda"))/255.
#   r = torch.Tensor(r).to(torch.device("cuda"))
#   for k in range(5000):
#     loss, _ = ae.training_step(images, r)
#     print("Rec error: {}".format(loss))
#     print()
#
#   uu = 11
#   _, a, y = ae.forward(images[uu:uu+1], r[uu:uu+1])
#   print()
#   print(a)
#
#
#   import matplotlib.pyplot as plt
#   y = y.permute(0, 2, 3, 1)[0]
#   plt.figure()
#
#   img = np.array(y.cpu().data*255)
#   img = img.astype('int32')
#   plt.imshow(img)
#   plt.show()
#
# #1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
# #         1.0000, 1.0000, 1.0000, 0.2844, 1.0000, 0.6574, 1.0000, 1.0000, 1.0000,
# #         1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
# #         1.0000, 0.9981, 1.0000, 1.0000, 1.0000


from scripts import parameters
import gym, torch
import gym_billiard
import matplotlib.pyplot as plt
import numpy as np
from core.metrics import ae, rnd
from core.qd import population, agents
from core.utils import utils
import os
if __name__ == "__main__":

  # Parameters
  # -----------------------------------------------
  load_path = '/home/giuseppe/src/rnd_qd/experiments/TEST_AE_norm/7'

  params = parameters.Params()
  params.load(os.path.join(load_path, 'params.json'))

  env = gym.make(params.env_tag)
  env.reset()
  # -----------------------------------------------

  # Possible targets
  # -----------------------------------------------
  with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
    x = np.load(f)
  # -----------------------------------------------

  # Load metric
  # -----------------------------------------------
  print('Loading metric...')
  device = torch.device('cpu')

  if params.metric == 'AE':
    selector = ae.ConvAutoEncoder(device=device, encoding_shape=params.feature_size)
    selector.load(os.path.join(load_path, 'models/ckpt/ckpt_ae.pth'))
  elif params.metric == 'RND':
    selector = rnd.RND(params.feature_size)
    selector.load(os.path.join(load_path, 'models/ckpt_rnd.pth'))
  else:
    raise ValueError('Wrong metric selected: {}'.format(params.metric))

  x[x==180] = 255
  images = torch.Tensor(x).permute(0, 3, 1, 2) / 255.

  E = torch.mean(images, dim=0)
  V = torch.var(images, dim=0)
  print(E.shape)
  print(V.shape)

  images = (images - E)#    bn(images)

  test_lim = 10

  aa = []
  for uu in range(test_lim):
    _, a, y = selector(images[uu:uu+1])
    print()
    print(a)
    aa.append(a.cpu().detach().numpy())

  aa_diff = np.zeros((test_lim,test_lim))



  for ii in range(test_lim):
    for jj in range(test_lim):
      d = aa[ii] - aa[jj]
      aa_diff[ii,jj] = np.sqrt(np.sum(d*d))

  print(aa_diff)

  import matplotlib.pyplot as plt


  fig, ax = plt.subplots(1,test_lim)
  for uu in range(test_lim):
    ax[uu].imshow(x[uu])

  plt.show()

  uu = 2
  _, a, y = selector(images[uu:uu + 1])
  y = y.permute(0, 2, 3, 1)[0]
  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(images.permute(0,2,3,1).cpu().data[uu])

  img = np.array(y.cpu().data*255)
  img = img.astype('int32')
  ax[1].imshow(img)
  plt.show()

