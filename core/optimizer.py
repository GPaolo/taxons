import numpy as np

class Optimizer(object):
  def __init__(self, env, pop, metric):
    self.env = env
    self.pop = pop
    self.metric = metric

