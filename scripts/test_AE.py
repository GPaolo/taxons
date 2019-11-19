# # Created by Giuseppe Paolo
# # Date: 20/02/19


from scripts import parameters
import torch
import matplotlib.pyplot as plt
import numpy as np
from core.metrics import ae, rnd
from core.utils import utils
import gym, gym_billiard

import os

class Tester(object):
  def __init__(self, path, device):
    self.params = parameters.Params()
    self.params.load(os.path.join(path, 'params.json'))

    print('Loading metric...')
    self.selector = ae.ConvAE(device=device, encoding_shape=self.params.feature_size, learning_rate=0.0001, beta=1)

    self.selector.load(os.path.join(path, 'models/ckpt/ckpt_ae.pth'))
    self.selector.training = False

  def __call__(self, data):
    return self.selector(data)

if __name__ == "__main__":

  TEST_TRAINED = True

  # Parameters
  # -----------------------------------------------
  seed = 3
  name = 'Billiard_AE'
  device = torch.device('cpu')

  load_path_AES = '/home/giuseppe/src/taxons/experiments/{}_Surprise/{}'.format(name, seed)
  load_path_AEN = '/home/giuseppe/src/taxons/experiments/{}_Novelty/{}'.format(name, seed)
  load_path_Mixed = '/home/giuseppe/src/taxons/experiments/{}_Mixed/{}'.format(name, seed)
  load_path_NT = '/home/giuseppe/src/taxons/experiments/{}_NoTrain/{}'.format(name, seed)

  env_tag = "Billiard-v0"
  number_of_samples = 10
  # -----------------------------------------------

  # -----------------------------------------------
  AEN = Tester(load_path_AEN, device)
  AES = Tester(load_path_AES, device)
  Mixed = Tester(load_path_Mixed, device)
  NT = Tester(load_path_NT, device)
  # -----------------------------------------------

  # Test samples
  # -----------------------------------------------
  x_test = []
  env = gym.make(env_tag)
  if "Billiard" in env_tag:
    env.env.params.RANDOM_BALL_INIT_POSE = True
    env.env.params.RANDOM_ARM_INIT_POSE = True
  elif "Ant" in env_tag:
    env.render()

  for k in range(50):
    env.reset()
    if "Ant" in env_tag:
      for step in range(300):
        env.step(env.action_space.sample())
        CoM = np.array([env.env.data.qpos[:2]])
        if np.any(np.abs(CoM) >= np.array([3, 3])):
          break
    tmp = env.render(mode='rgb_array')
    x_test.append(tmp)
  x_test = np.stack(x_test) / np.max((np.max(x_test), 1))
  images_test = torch.Tensor(x_test).permute(0, 3, 1, 2).to(device)
  # -----------------------------------------------

  # Test
  # -----------------------------------------------
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(1, number_of_samples)
  for k in range(number_of_samples):
    ax[k].imshow(x_test[k, :, :])
  plt.show()

  for k in range(number_of_samples):
    N_loss, N_f, N_y = AEN(images_test[k:k+1])
    S_loss, S_f, S_y = AES(images_test[k:k+1])
    M_loss, M_f, M_y = Mixed(images_test[k:k+1])
    NT_loss, NT_f, NT_y = NT(images_test[k:k+1])

    print()
    print('Reconstruction Error: \n'
          'AEN {} \nAES {} \nMixed {} \nNT {}'.format(N_loss.cpu().data, S_loss.cpu().data, M_loss.cpu().data, NT_loss.cpu().data))

    N_y = N_y.permute(0, 2, 3, 1)[0]
    S_y = S_y.permute(0, 2, 3, 1)[0]
    M_y = M_y.permute(0, 2, 3, 1)[0]
    NT_y = NT_y.permute(0, 2, 3, 1)[0]

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(images_test.permute(0,2,3,1).cpu().data[k])
    ax[0, 0].set_title('Original')

    subs = AEN.selector.subsample(images_test[k:k+1])
    subs = subs.permute(0, 2, 3, 1)[0]
    ax[0, 1].imshow(np.array(subs.cpu().data))
    ax[0, 1].set_title('Subsampled')

    img = np.array(N_y.cpu().data * 255)
    img = img.astype('int32')
    ax[0, 2].imshow(img)
    ax[0, 2].set_title('AEN')

    img = np.array(S_y.cpu().data * 255)
    img = img.astype('int32')
    ax[1, 0].imshow(img)
    ax[1, 0].set_title('AES')

    img = np.array(M_y.cpu().data * 255)
    img = img.astype('int32')
    ax[1, 1].imshow(img)
    ax[1, 1].set_title('Mixed')

    img = np.array(NT_y.cpu().data * 255)
    img = img.astype('int32')
    ax[1, 2].imshow(img)
    ax[1, 2].set_title('NT')

    plt.show()
  # -----------------------------------------------


