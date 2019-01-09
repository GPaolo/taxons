import numpy as np


class FCLayer(object):
  def __init__(self, input, output, name=None):
    super().__init__()
    self.w = np.random.rand(input, output)
    self.bias = np.random.rand(1, output)
    if name is None:
      self.name = 'fc'
    else:
      self.name = name

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    assert np.shape(x)[0] == 1, 'Wrong input shape. Needs to be [1, n]'
    return np.matmul(x, self.w) + self.bias

  @property
  def params(self):
    return self.w, self.bias

  @property
  def show(self):
    print('Layer {}:'.format(self.name))
    print('Weights {}'.format(self.w))
    print('Bias {}'.format(self.bias))