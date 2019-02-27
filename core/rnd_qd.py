import numpy as np
from core.metrics import rnd, ae
from core.qd import population, agents
from core.utils import utils
from core.utils import optimizer
import gym, torch
import gym_billiard
import os, threading
import matplotlib
from tensorboardX import SummaryWriter
env_tag = 'Billiard-v0'
# env_tag = 'MountainCarContinuous-v0'


class RndQD(object):

  def __init__(self, env, agents_shapes, bs_shape, pop_size, save_path, agent_name, use_novelty=True, use_archive=False, gpu=False):
    '''

    :param env: Environment in which we act
    :param bs_shape: dimension of the behavious space
    '''
    self.pop_size = pop_size
    self.use_novelty = use_novelty
    self.parameters = None
    self.env = env
    self.save_path = save_path
    self.agents_shapes = agents_shapes
    self.agent_name = agent_name

    self.writer = SummaryWriter(self.save_path)
    self.metric_update_steps = 0
    self.metric_update_single_agent = True

    if self.agent_name == 'Neural':
      agent_type = agents.FFNeuralAgent
    elif self.agent_name == 'DMP':
      agent_type = agents.DMPAgent

    self.population = population.Population(agent=agent_type,
                                            shapes=self.agents_shapes,
                                            pop_size=self.pop_size)
    self.archive = None
    if use_archive:
      self.archive = population.Population(agent=agent_type,
                                           shapes=self.agents_shapes,
                                           pop_size=0)

    if gpu:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device('cpu')
    if self.use_novelty:
      self.metric = ae.AutoEncoder(device=self.device, encoding_shape=bs_shape)
    else:
      self.metric = None

    self.opt = optimizer.NoveltyOptimizer(self.population, archive=self.archive)
    self.cumulated_state = []

    self.END = False
    self.thread = threading.Thread(target=self._control_interface)
    self.thread.start()

  def _control_interface(self):
    print('If you want to show the progress, press s.')
    print('If you want to stop training, press q.')
    matplotlib.use('agg')
    while True:
      action = input(' ')
      if action == 's':
        try:
          if self.archive is not None:
            bs_points = np.concatenate(self.archive['bs'].values)
          else:
            bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
          utils.show(bs_points, filepath=self.save_path)
        except:
          print('Cannot show progress now.')
      elif action == 'q':
        print('Quitting training...')
        self.END = True

  # TODO make this run in parallel
  def evaluate_agent(self, agent):
    '''
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    '''
    done = False
    cumulated_reward = 0

    obs = utils.obs_formatting(env_tag, self.env.reset())
    t = 0
    while not done:
      if self.agent_name == 'Neural':
        agent_input = obs
      elif self.agent_name == 'DMP':
        agent_input = t

      action = utils.action_formatting(env_tag, agent['agent'](agent_input))

      obs, reward, done, info = self.env.step(action)
      obs = utils.obs_formatting(env_tag, obs)
      t += 1
      cumulated_reward += reward

    surprise = 0
    if self.use_novelty:
      state = self.env.render(rendered=False)
      state = torch.Tensor(state).permute(2,0,1).to(self.device)

      if self.metric_update_single_agent:
        surprise = self.metric.training_step(state.unsqueeze(0))# Input Dimensions need to be [1, input_dim]
        self.metric_update_steps += 1
        self.writer.add_scalar('novelty', surprise, self.metric_update_steps)
      else:
        self.cumulated_state.append(state)
        surprise = self.metric(state.unsqueeze(0))
      surprise = surprise.cpu().data.numpy()
      # self.cumulated_state.append(state) # Append here all the states

    agent['bs'] = np.array([[obs[0][0], obs[0][1]]])
    agent['surprise'] = surprise
    agent['reward'] = cumulated_reward

  def update_metric(self):
    '''
    This function uses the cumulated state to update the metrics parameters and then empties the cumulated_state
    :return:
    '''
    self.cumulated_state = torch.stack(self.cumulated_state).to(self.device)
    cum_surprise = self.metric.training_step(self.cumulated_state)

    self.metric_update_steps += 1
    self.writer.add_scalar('novelty', cum_surprise, self.metric_update_steps)

    self.cumulated_state = []
    return cum_surprise

  def train(self, steps=10000):
    '''
    This function trains the agents and the RND
    :param steps: number of update steps (or generations)
    :return:
    '''
    self.elapsed_gen = 0
    for self.elapsed_gen in range(steps):
      cs = 0
      max_rew = -np.inf
      for a in self.population:
        self.evaluate_agent(a)
        cs += a['surprise']
        if max_rew < a['reward']:
          max_rew = a['reward']

      if not self.metric_update_single_agent:
        self.update_metric()

      self.opt.step()

      self.writer.add_scalar('Archive_size', self.archive.size, self.elapsed_gen)
      self.writer.add_scalar('Avg_generation_novelty', cs/self.pop_size)

      if self.elapsed_gen % 10 == 0:
        print('Generation {}'.format(self.elapsed_gen))
        if self.archive is not None:
          print('Archive size {}'.format(self.archive.size))
        print('Average generation surprise {}'.format(cs/self.pop_size))
        print('Max reward {}'.format(max_rew))
        print()

      if self.END:
        print('Quitting.')
        break

  def save(self):
    print('Saving...')
    if not os.path.exists(self.save_path):
      try:
        os.mkdir(os.path.abspath(self.save_path))
      except:
        print('Cannot create save folder.')
    self.population.save_pop(self.save_path, 'pop')
    self.archive.save_pop(self.save_path, 'archive')
    self.metric.save(self.save_path)

    self.writer.export_scalars_to_json(os.path.join(self.save_path, "scalars_log.json"))
    self.writer.close()

    print('Done')

if __name__ == '__main__':
  env = gym.make(env_tag)

  env.seed()
  np.random.seed()
  torch.initial_seed()

  rnd_qd = RndQD(env, action_shape=2, obs_shape=6, bs_shape=64, pop_size=100, use_novelty=True, use_archive=True, gpu=True)
  try:
    rnd_qd.train(500)
  except KeyboardInterrupt:
    print('User Interruption.')

  rnd_qd.save('RND_QD_{}'.format(rnd_qd.elapsed_gen))

  if rnd_qd.archive is None:
    pop = rnd_qd.population
  else:
    pop = rnd_qd.archive
  print('Total generations: {}'.format(rnd_qd.elapsed_gen))
  print('Archive length {}'.format(pop.size))

  if rnd_qd.archive is not None:
    bs_points = np.concatenate(rnd_qd.archive['bs'].values)
  else:
    bs_points = np.concatenate([a['bs'] for a in rnd_qd.population if a['bs'] is not None])
  utils.show(bs_points, 'RNDQD_{}_{}'.format(rnd_qd.elapsed_gen, env_tag))

  print('Testing result according to best reward.')
  rewards = pop['reward'].sort_values(ascending=False)
  for idx in range(pop.size):
    tested = pop[rewards.iloc[idx:idx+1].index.values[0]]
    print()
    print('Testing agent {} with reward {}'.format(tested['name'], tested['reward']))
    done = False
    ts = 0
    obs = utils.obs_formatting(env_tag, rnd_qd.env.reset())
    while not done and ts < 3000:
      rnd_qd.env.render()
      action = utils.action_formatting(env_tag, tested['agent'](obs))
      obs, reward, done, info = rnd_qd.env.step(action)
      obs = utils.obs_formatting(env_tag, obs)
      ts += 1

