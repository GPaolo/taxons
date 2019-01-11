from rnd_qd.core.qd import agents
import numpy as np

np.random.seed(7)

def test_base_agent():
  agent = agents.BaseAgent(mutation_distr=None)
  assert np.isclose(agent.mutation_operator(), 0.33810514076007125), 'Default mutation operator does not work'

  agent = agents.BaseAgent(mutation_distr=np.random.randint)
  assert  agent.mutation_operator(10) == 9, 'Cannot pass new mutation operator.'

def test_neural_agent():
  shapes = {'input_shape': 3, 'output_shape':2}
  agent = agents.FFNeuralAgent(shapes)

  assert len(agent.genome) == 4, 'Wrong genome len.'

  x = np.ones((1,3))
  try:
    result = agent(x)
  except:
    raise Exception('Call function not working.')

  assert np.shape(result) == (1,2), 'Wrong output shape.'
  assert np.allclose(result, [[-0.70445828, -0.94089774]]), 'Wrong output.'


