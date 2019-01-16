from rnd_qd.core.qd import agents
import numpy as np

np.random.seed(7)

def test_base_agent():
  agent = agents.BaseAgent(mutation_distr=None)
  try:
    agent.mutate()
  except NotImplementedError:
    pass

def test_neural_agent():
  shapes = {'input_shape': 3, 'output_shape':2}
  agent = agents.FFNeuralAgent(shapes)

  assert len(agent.genome) == 4, 'Wrong genome len.'

  x = np.ones((1,3))
  try:
    result = agent(x)
  except:
    raise Exception('Call function not working.')

  assert np.shape(result) == (1,2), "Wrong output shape."
  assert np.allclose(result, [[0.64340802, -0.43566705]]), 'Wrong output.'

def test_mutation_operator():
  shapes = {'input_shape': 3, 'output_shape': 2}
  agent = agents.FFNeuralAgent(shapes)

  assert np.isclose(agent.mutation_operator(), -0.023661300383801844), "Default mutation operator does not work"

  agent = agents.FFNeuralAgent(shapes, mutation_distr=np.random.randint)
  assert agent.mutation_operator(10) == 2, 'Cannot pass new mutation operator.'

