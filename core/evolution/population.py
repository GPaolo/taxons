from core.evolution.agents import *
import pandas as pd
import os
import pickle as pkl

class Population(object):
  """
  Population class. The new generation is just the mutation of the best elements that substitutes the worst.
  The criteria for the best is given by the metric, and is calculated outside.
  """
  # ---------------------------------
  def __init__(self, shapes, agent=BaseAgent, pop_size=10, max_len=None):
    """
    Constructors
    :param shapes: Parameters for the agents
    :param agent: Agent type
    :param pop_size: Size of the initial population
    :param max_len: Maximum length of the pop in case we use a growing population
    """
    self.pop = pd.DataFrame(columns=['agent', 'reward', 'surprise', 'best', 'bs', 'name', 'novelty', 'features'])
    self.agent_class = agent
    self.shapes = shapes
    self.max_len = max_len
    self.avg_surprise = 0
    self.agent_name = 0

    for i in range(pop_size):
      self.add()
  # ---------------------------------

  # These functions allow to work with the pop as a list
  # ---------------------------------
  def __iter__(self):
    """
    Allows to directly iterate the pop.
    :return:
    """
    self._iter_idx = 0
    return self

  def __next__(self):
    """
    During iteration returns the next element of the iterator
    :return:
    """
    if self._iter_idx < self.size:
      x = self.pop.loc[self._iter_idx]
      self._iter_idx += 1
    else:
      raise StopIteration
    return x

  def __getitem__(self, item):
    """
    Returns the asked item
    :param item: item to return.
    :return: If item is a string returns a column of the dataframe. If it is an integer returns the corresponding agent
    """
    if type(item) is str:
      return self.pop[item]
    return self.pop.iloc[item]

  def __setitem__(self, key, value):
    """
    Set the agent in position key with the ones passed as value
    :param key: Position of the agent to set
    :param value: New agent to set
    :return:
    """
    assert key < self.size and key > -self.size-1, 'Index out of range'
    self.pop.iloc[key] = value

  def __len__(self):
    """
    Returns the length of the population
    """
    return self.size

  @property
  def size(self):
    """
    Size of the population
    """
    return len(self.pop)
  # ---------------------------------

  # ---------------------------------
  def add(self, agent=None):
    """
    Adds agent to the pop. If no agent is passed, a new agent is generated.
    :param agent: agent to add
    :return:
    """
    if agent is None:
      agent = {'agent': self.agent_class(self.shapes), 'reward': None, 'surprise': None, 'novelty': None,
               'best': False, 'bs':None, 'name':self.agent_name, 'features': None}
      self.agent_name += 1

    agent = pd.DataFrame([agent], columns=agent.keys()) # If an agent is given, it should already have a name
    self.pop = pd.concat([self.pop, agent], ignore_index=True, sort=True)
  # ---------------------------------

  # ---------------------------------
  def show(self):
    """
    Prints the population
    """
    for a in self:
      print(a)
  # ---------------------------------

  # ---------------------------------
  def copy(self, idx, with_data=False):
    """
    Returns a copy of the agent at position idx
    :param idx: Position of the agent to copy
    :param with_data: If true also copies all the data relative to the agent.
    :return: Copy of the agent
    """
    assert idx < self.size and idx > -self.size-1, 'Index out of range'
    agent = {'agent': self.agent_class(self.shapes), 'reward': None, 'surprise': None, 'novelty': None,
             'best': False, 'bs': None, 'name':self.agent_name, 'features': None}

    if with_data:
      for key in agent.keys(): # If copied with data we keep the original name
        agent[key] = deepcopy(self[idx][key])
    else:
      agent['agent'] = deepcopy(self[idx]['agent'])
      self.agent_name += 1 # if not with data, the agent is new, so we update the name

    agent = pd.DataFrame([agent], columns=agent.keys())
    return agent.iloc[0]
  # ---------------------------------

  # ---------------------------------
  def save_pop(self, filepath, name):
    """
    Saves the population as a .json file
    :param filepath:
    :param name: Name of the file where to save the pop
    """
    save_ckpt = {}

    save_ckpt['Agent Type'] = self.agent_class.__name__
    save_ckpt['Genome'] = {}

    for a in self:
      save_ckpt['Genome'][a['name']] = {'gen': a['agent'].genome, 'feat': a['features'], 'bs': a['bs']}
    try:
      with open(os.path.join(filepath, 'qd_{}.pkl'.format(name)), 'wb') as file:
        pkl.dump(save_ckpt, file)
    except Exception as e:
      print('Cannot Save {}.'.format(name))
      print('Exception {}'.format(e))
  # ---------------------------------

  # ---------------------------------
  def load_pop(self, filepath):
    """
    Loads a population from a given filepath
    :param filepath: Filepath from where to save the population
    """
    if not os.path.exists(filepath):
      print('File to load not found.')
      return

    print('Loading population from {}'.format(filepath))
    with open(filepath, 'rb') as file:
      ckpt = pkl.load(file)

    # Check if we are loading the right agent class
    assert ckpt['Agent Type'] == self.agent_class.__name__, "Wrong agent type. Saved {}, current {}".format(ckpt['Agent Type'], self.agent_class.__name__)

    # Creates empty pop
    del self.pop
    self.pop = pd.DataFrame(columns=['agent', 'reward', 'surprise', 'best', 'bs', 'name', 'novelty', 'features'])
    self.agent_name = 0

    # Start loading agents
    for agent_name in ckpt['Genome']:
      # Create empty agent
      agent = {'agent': self.agent_class(self.shapes), 'reward': None, 'surprise': None, 'novelty': None,
               'best': False, 'bs': None, 'name': agent_name, 'features': None}
      try:
        agent_genome = ckpt['Genome'][agent_name]['gen'] # Get genome
        agent['features'] = ckpt['Genome'][agent_name]['feat'] # Get features
        try:
          agent['bs'] = ckpt['Genome'][agent_name]['bs'] # Get ground truth BS
        except:
          print('Agents without bs')
      except:
        print('Agents without features!')
        agent_genome = ckpt['Genome'][agent_name] # Get genome

      # Check if genome is of the right size
      assert len(agent_genome) == len(agent['agent'].genome), 'Wrong genome length. Saved {}, current {}'.format(agent_genome, self[-1]['agent'].genome)
      agent['agent'].load_genome(agent_genome, agent_name) # Load genome to agent
      # Check that genome has been loaded properly
      for k in range(len(agent_genome)):
        try:
          for p in agent['agent'].genome[k]:
            assert np.all(agent['agent'].genome[k][p] == agent_genome[k][p]), 'Could not load {} of element {} in agent {}'.format(p, k, agent)
        except TypeError: #TODO this is because the action len is stored as a float in the list. Might have to put it into a dict so don't have to do the exception
          assert agent['agent'].genome[k] == agent_genome[k], 'Could not load action_len of element {} in agent {}'.format(p, k, agent)

      self.add(agent) # Add loaded agent to the population
    print("Done")
  # ---------------------------------
