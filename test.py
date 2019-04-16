# Created by Giuseppe Paolo
# Date: 20/02/19

import numpy as np

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


if __name__ == "__main__":
  neat = NEATAgent(3,4)
  print(neat.connection_matrix)
  neat._add_node()
  print(neat.connection_matrix)
