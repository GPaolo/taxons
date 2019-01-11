import numpy as np


class FCLayer(object):
  def __init__(self, input, output, name='fc'):
    super().__init__()
    self.w = np.random.randn(input, output) * 0.1
    self.bias = np.random.randn(1, output)
    self.name = name

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    assert np.shape(x)[0] == 1, 'Wrong input shape. Needs to be [1, n]. Is {}'.format(np.shape(x))
    return np.matmul(x, self.w) + self.bias

  @property
  def params(self):
    return self.w, self.bias

  @property
  def show(self):
    print('Layer {}:'.format(self.name))
    print('Weights {}'.format(self.w))
    print('Bias {}'.format(self.bias))