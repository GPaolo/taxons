# # Created by Giuseppe Paolo
# # Date: 20/02/19


from scripts import parameters
import torch
import matplotlib.pyplot as plt
import numpy as np
from core.metrics import ae, rnd
from core.utils import utils

import os
if __name__ == "__main__":

  TEST_TRAINED = False

  # Parameters
  # -----------------------------------------------
  load_path = '/home/giuseppe/src/rnd_qd/experiments/AA/7'

  params = parameters.Params()
  params.load(os.path.join(load_path, 'params.json'))

#  env = gym.make(params.env_tag)
#  env.reset()
  # -----------------------------------------------

  # Load metric
  # -----------------------------------------------
  print('Loading metric...')
  device = torch.device('cuda')
  selector = ae.ConvAutoEncoder(device=device, encoding_shape=params.feature_size, learning_rate=0.0001)
  # -----------------------------------------------

  # Possible targets
  # -----------------------------------------------
  with open('/home/giuseppe/src/rnd_qd/train_img.npy', 'rb') as f:
    x = np.load(f)

  images = torch.Tensor(x).permute(0, 3, 1, 2).to(device)/255.
  # -----------------------------------------------

  # Train/Load trained
  # -----------------------------------------------
  if not TEST_TRAINED:
    total_epochs = 500
    batches = utils.split_array(images, batch_size=64)
    loss = 0
    for k in range(total_epochs):
      for data in batches:
        loss, f, y = selector.training_step(data)
      print('Epoch {} - Loss {}'.format(k, loss))
      # if loss.cpu().data < 0.001:
      #   break
    selector.save(load_path)
  # -----------------------------------------------
  else:
    selector.load(os.path.join(load_path, 'models/ckpt_ae.pth'))
  # -----------------------------------------------


  # Load test samples
  # -----------------------------------------------
  with open('/home/giuseppe/src/rnd_qd/input_img.npy', 'rb') as f:
    x_test = np.load(f)
  images_test = torch.Tensor(x_test).permute(0, 3, 1, 2).to(device)
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

  uu = 2
  _, a, y = selector(images_test[uu:uu + 1])
  y = y.permute(0, 2, 3, 1)[0]
  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(images_test.permute(0,2,3,1).cpu().data[uu])

  img = np.array(y.cpu().data*255)
  img = img.astype('int32')
  ax[1].imshow(img)
  plt.show()
  # -----------------------------------------------


