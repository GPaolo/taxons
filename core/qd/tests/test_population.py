from rnd_qd.core.qd import population
from copy import deepcopy

def test_iter():
  pop = population.Population()

  for k in pop:
    print(k)

def test_get_item():
  pop = population.Population()
  a = pop[3]
  assert pop.pop[3]['agent'] == a['agent'], 'Got wrong agent.'

def test_set_item():
  pop = population.Population()
  a = deepcopy(pop[3])
  assert not pop.pop[3]['agent'] == a['agent'], 'Could not deepcopy the agent.'
  pop[3] = a
  assert pop.pop[3]['agent'] == a['agent'], 'Could not set agent.'

def test_append():
  pop = population.Population()
  len_pop = len(pop)
  a = deepcopy(pop[3])
  pop._append(a)
  assert len_pop + 1 == len(pop), 'Could not append agent.'

def test_add():
  pop = population.Population()
  len_pop = len(pop)
  pop.add()
  assert len_pop + 1 == len(pop), 'Could not add base agent.'
  a = deepcopy(pop[3])
  pop.add(a)
  assert len_pop + 2 == len(pop), 'Could not add copy of agent.'
  assert pop.pop[-1]['agent'] == a['agent'], 'Added wrong agent'
