# Created by Giuseppe Paolo
# Date: 20/02/19

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import gym_billiard
import sys
from core.qd import population, agents
from core.utils import utils

class AutoEncoder(nn.Module):

  def __init__(self):
    super(AutoEncoder, self).__init__()

    device = None
    if device is not None:
      self.device = device
    else:
      self.device = torch.device("cpu")

    self.subsample = nn.MaxPool2d(8).to(self.device)  # 600 -> 75

    # self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2), nn.ReLU(), # 75 -> 36
    #                             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2), nn.ReLU(), # 36 -> 17
    #                             nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1), nn.ReLU()).to(self.device) # 17 -> 15
    #
    # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3, stride=1), nn.ReLU(), # 8 -> 17
    #                             nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2), nn.ReLU(), # 17 -> 36
    #                             nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=5, stride=2), nn.ReLU()).to(self.device) # 36 -> 75
    #
    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2), nn.LeakyReLU(),  # 75 -> 35
                                 nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=3), nn.LeakyReLU()).to(self.device)  # 35 -> 11
    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=5, stride=3), nn.LeakyReLU(),
                                 nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=7, stride=2), nn.ReLU()).to(self.device)

    self.encoder_ff = nn.Sequential(nn.Linear(484, 32), nn.LeakyReLU()).to(self.device)
    self.decoder_ff = nn.Sequential(nn.Linear(32, 484), nn.LeakyReLU()).to(self.device)

    self.criterion = nn.MSELoss()
    self.learning_rate = 0.001
    self.zero_grad()
    self.optimizer = optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)

    self.to(self.device)
    self.criterion.to(self.device)


  def forward(self, x):
    x = self.subsample(x)
    feat = self.encoder(x)

    shape = feat.shape
    feat = feat.view(-1, 484)

    feat = self.encoder_ff(feat)
    y = self.decoder_ff(feat)
    y = y.view(shape)

    y = self.decoder(y)

    return y, feat

  def train_ae(self, x):
    self.optimizer.zero_grad()

    y = self.forward(x)
    x = self.subsample(x)
    loss = self.criterion(x, y)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
    self.optimizer.step()
    self.scheduler.step()
    return loss

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

if __name__ == '__main__':
  env_tag = 'Billiard-v0'
  env = gym.make(env_tag)
  #
  env.reset()
  #state = env.render(rendered=False)
  #
  # with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
  #  tens = np.load(f)
  #
  #test = np.append(tens, np.expand_dims(state, 0), 0)
  #with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'wb') as f:
  #  test.dump(f)
  #print(test.shape)

  #from tensorboardX import SummaryWriter
  #
  import matplotlib.pyplot as plt
  with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
    x = np.load(f)
  # print(x.shape)
  #
  # norm = False
  # factor = 1
  # if norm:
  #   factor = 255
  #
  # net = AutoEncoder()
  # net.load('/home/giuseppe/src/rnd_qd/ckpt_ae.pth')
  #
  #
  # x = torch.Tensor(x/factor).permute(3, 0, 1, 2).to(net.device)
  # train = x[:, 15:]
  # test = x[:, :15]
  # print(test.shape)
  # test.to(net.device)
  # train.to(net.device)
  #
  # a = net.subsample(x)
  # a = a.permute(1,2,3,0)
  # fig, ax = plt.subplots(2)
  # a = a.cpu().data.numpy()
  # if not norm:
  #   a = a.astype(np.int)
  #
  # ax[0].imshow(a[0])
  # plt.imshow(a[0])
  # ax[1].imshow(a[1])
  # plt.imshow(a[1])
  # plt.show()



  #a = net(x.unsqueeze(0))
  #print('Starting training')
  #for k in range(5000):
  #  loss = net.train_ae(train.unsqueeze(0))
  #  if k%100 == 0:
  #    print('Loss at {}: {}'.format(k, loss))
  #  writer.add_scalar('loss', loss, k)

  # print(x.shape)
  # fig, ax = plt.subplots(7, 3)

  #writer.export_scalars_to_json("./all_scalars.json")
  #writer.close()

  # b = net(test.unsqueeze(0))
  # b = b[0].permute(1, 2, 3, 0)
  #
  # for i in range(7):
  #   for j in range(3):
  #     k = i+j
  #     #b = net(x[k:k+1])
  #     a = x[k]
  #     #a = a.permute(1, 2, 0)
  #     a = a.cpu().data.numpy()
  #     if not norm:
  #       a = a.astype(np.int)
  #
      # ax[i, j].imshow(a)
      # plt.imshow(a)
  # plt.show()

  # fig, ax = plt.subplots(1)
  #b = net(test)
  #a = b[0]
  #a = a.permute(1, 2, 0)
  #a = a.cpu().data.numpy()
  #if not norm:
  #  a = a.astype(np.int)
  # ax.imshow(x[20])
  #
  #test = test[0].permute(1,2,0)
  #test = test.cpu().data.numpy()
  #ax[1].imshow(test)
  #
  #k = net.subsample(x[15:16])
  #ll = net.criterion(b, k)
  #print(ll)
  # plt.show()
  ae_model = AutoEncoder()
  ae_model.load('/home/giuseppe/src/rnd_qd/experiments/ae_deep_novelty_reupdated_fea/models/ckpt_ae.pth')

  # archive = population.Population(agent=agents.DMPAgent, pop_size=0, shapes={'dof': 2, 'degree': 5})
  # archive.load_pop('/home/giuseppe/src/rnd_qd/experiments/ae_deep_novelty_reupdated_fea/models/qd_archive.pkl')

  pop = population.Population(agent=agents.DMPAgent, pop_size=0, shapes={'dof': 2, 'degree': 5})
  pop.load_pop('/home/giuseppe/src/rnd_qd/experiments/ae_deep_novelty_reupdated_fea/models/qd_pop.pkl')

  # Evaluate agents bs points
  # ----------------------------------------------------------------
  i = 0
  for agent in pop:
    if i % 50 == 0:
      print('Evaluating agent {}'.format(i))
    done = False
    obs = utils.obs_formatting(env_tag, env.reset())
    t = 0
    while not done:
      agent_input = t
      action = utils.action_formatting(env_tag, agent['agent'](agent_input))

      obs, reward, done, info = env.step(action)
      obs = utils.obs_formatting(env_tag, obs)
      t += 1

    state = env.render(rendered=False)
    state = torch.Tensor(state).permute(2, 0, 1).unsqueeze(0)
    _, bs_point = ae_model(state)
    bs_point = bs_point .flatten().cpu().data.numpy()

    agent['features'] = [bs_point]
    i += 1
  # ----------------------------------------------------------------

  # Calculate New point bs
  a = torch.Tensor(x[19]).permute(2, 0, 1).unsqueeze(0)
  _, bs_point = ae_model(a)
  bs_point = bs_point.flatten().cpu().data.numpy()

  # Get N closest agents
  # ----------------------------------------------------------------
  bs_space = np.stack([a[0] for a in pop['features'].values])

  # Get distances
  diff = np.atleast_2d(bs_space - bs_point)
  dists = np.sqrt(np.sum(diff * diff, axis=1))
  k = 5
  if len(dists) <= k:  # Should never happen
    idx = list(range(len(dists)))
    k = len(idx)
  else:
    idx = np.argpartition(dists, k)  # Get 15 nearest neighs

  mean_k_dist = np.mean(dists[idx[:k]])

  def get_scaling(k):
    scale = np.array(list(range(k)))/k
    scale = scale / np.sum(scale)
    return scale








