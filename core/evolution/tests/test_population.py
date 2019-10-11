from core.rnd_qd import population


def test_iter():
  pop = population.Population()

  for k in pop:
    k['best'] = True

  for i in range(pop.size):
    assert pop.pop.iloc[i]['best'], 'Could not iterate properly. '

def test_get_item():
  pop = population.Population()
  a = pop[3]
  assert pop.pop.loc[3]['agent'] == a['agent'], 'Got wrong agent.'

def test_set_item():
  pop = population.Population()
  a = pop.copy(3)
  assert not pop.pop.loc[3]['agent'] == a['agent'], 'Could not deepcopy the agent.'
  pop[3] = a
  assert pop.pop.loc[3]['agent'] == a['agent'], 'Could not set agent.'

def test_add():
  pop = population.Population()
  len_pop = len(pop)
  pop.add()
  assert len_pop + 1 == len(pop), 'Could not add base agent.'
  a = pop.copy(3)
  pop.add(a)
  assert len_pop + 2 == len(pop), 'Could not add copy of agent.'
  assert pop[-1]['agent'] == a['agent'], 'Added wrong agent'
