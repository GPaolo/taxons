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
if __name__ == "__main__":

  TEST_TRAINED = True

  # Parameters
  # -----------------------------------------------
  load_path = '/home/giuseppe/src/rnd_qd/experiments/Ball.2.Surprise/7'

  params = parameters.Params()
  params.load(os.path.join(load_path, 'params.json'))

#  env = gym.make(params.env_tag)
#  env.reset()
  # -----------------------------------------------

  # Load metric
  # -----------------------------------------------
  print('Loading metric...')
  device = torch.device('cpu')
  selector = ae.AutoEncoder(device=device, encoding_shape=params.feature_size, learning_rate=0.0001, beta=1)
  # -----------------------------------------------

  # Possible targets
  # -----------------------------------------------
  with open('/home/giuseppe/src/rnd_qd/train_img.npy', 'rb') as f:
    x = np.load(f)
  # -----------------------------------------------

  # Train/Load trained
  # -----------------------------------------------
  if not TEST_TRAINED:
    total_epochs = 3
    batches = utils.split_array(x, batch_size=100)
    loss = 0
    for k in range(total_epochs):
      for data in batches:
        images = torch.Tensor(data).permute(0, 3, 1, 2).to(device)/np.max(data)
        loss, f, y = selector.training_step(images)
      print('Epoch {} - Loss {}'.format(k, loss))
      # if loss.cpu().data < 0.001:
      #   break
    selector.save(load_path)
  # -----------------------------------------------
  else:
    selector.load(os.path.join(load_path, 'models/ckpt/ckpt_ae.pth'))
  selector.training = False
  # -----------------------------------------------


  # Load test samples
  # -----------------------------------------------
  # with open('/home/giuseppe/src/rnd_qd/test_img.npy', 'rb') as f:
  #   x_test = np.load(f)
  # images_test = torch.Tensor(x_test).permute(0, 3, 1, 2).to(device)/np.max(x_test)
  x_test = []
  env = gym.make("Billiard-v0")
  env.env.params.RANDOM_BALL_INIT_POSE = True
  env.env.params.RANDOM_ARM_INIT_POSE = True
  for k in range(50):
    env.reset()
    x_test.append(env.render(mode='rgb_array'))
  x_test = np.stack(x_test)
  images_test = torch.Tensor(x_test).permute(0, 3, 1, 2).to(device) / np.max(x_test)
  # -----------------------------------------------

  # Test
  # -----------------------------------------------
  test_lim = 10

  aa = []
  for k in range(test_lim):
    loss, a, y = selector(images_test[k:k+1])
    print()
    print('Reconstruction Error {}'.format(loss.cpu().data))
    print('Features {}'.format(a.cpu().detach().numpy()))
    aa.append(a.cpu().detach().numpy())

  aa_diff = np.zeros((test_lim,test_lim))

  for ii in range(test_lim):
    for jj in range(test_lim):
      d = aa[ii] - aa[jj]
      aa_diff[ii,jj] = np.sqrt(np.sum(d*d))

  print(aa_diff)

  import matplotlib.pyplot as plt


  fig, ax = plt.subplots(1, test_lim)
  for k in range(test_lim):
    ax[k].imshow(x_test[k, :, :])

  plt.show()

  for uu in range(min(20, len(images_test))):
    error, a, y = selector(images_test[uu:uu + 1])
    y = y.permute(0, 2, 3, 1)[0]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images_test.permute(0,2,3,1).cpu().data[uu])

    img = np.array(y.cpu().data*255)
    img = img.astype('int32')
    ax[1].imshow(img)
    print("Rec Error: {}".format(error.cpu().data))
    plt.show()
  # -----------------------------------------------


