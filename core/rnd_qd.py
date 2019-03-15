import numpy as np
from core.metrics import rnd, ae
from core.qd import population, agents
from core.utils import utils
from core.utils import optimizer
import gym, torch
import gym_billiard
import os, threading, sys, traceback
import matplotlib
from tensorboardX import SummaryWriter
env_tag = 'Billiard-v0'
# env_tag = 'MountainCarContinuous-v0'


class RndQD(object):

  def __init__(self, env, parameters):
    """
    :param env: Environment in which we act
    :param parameters: Parameters to use
    """
    self.params = parameters
    self.pop_size = self.params.pop_size
    self.env = env
    self.save_path = self.params.save_path
    self.agents_shapes = self.params.agent_shapes
    self.agent_name = self.params.qd_agent

    self.writer = SummaryWriter(self.save_path)
    self.metric_update_steps = 0
    self.metric_update_single_agent = self.params.per_agent_update

    if self.agent_name == 'Neural':
      agent_type = agents.FFNeuralAgent
    elif self.agent_name == 'DMP':
      agent_type = agents.DMPAgent

    self.population = population.Population(agent=agent_type,
                                            shapes=self.agents_shapes,
                                            pop_size=self.pop_size)
    self.archive = None
    if self.params.use_archive:
      self.archive = population.Population(agent=agent_type,
                                           shapes=self.agents_shapes,
                                           pop_size=0)

    if self.params.gpu:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device('cpu')

    if self.params.metric == 'AE':
      self.metric = ae.AutoEncoder(device=self.device, learning_rate=self.params.learning_rate, encoding_shape=self.params.feature_size)
    else:
      self.metric = rnd.RND(device=self.device, learning_rate=self.params.learning_rate, encoding_shape=self.params.feature_size)

    self.opt = self.params.optimizer(self.population, archive=self.archive)
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
        except BaseException as e:
          ex_type, ex_value, ex_traceback = sys.exc_info()
          trace_back = traceback.extract_tb(ex_traceback)
          stack_trace = list()
          for trace in trace_back:
            stack_trace.append(
              "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
          print('Cannot show progress due to {}: {}'.format(ex_type.__name__, ex_value))
          print(stack_trace[0])
      elif action == 'q':
        print('Quitting training...')
        self.END = True

  # TODO make this run in parallel
  def evaluate_agent(self, agent):
    """
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    """
    done = False
    cumulated_reward = 0
    state = []

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

      # Do not save every frame, but only once every N frames
      if t % self.params.state_recording_interval == 0:
        state.append(self.env.render(rendered=False))

    state = self.metric.subsample(torch.Tensor(np.stack(state)).permute(3, 0, 1, 2).unsqueeze(0))

    if self.metric_update_single_agent:
      surprise, features = self.metric.training_step(state.to(self.device))  # Input Dimensions need to be [1, input_dim]
      self.metric_update_steps += 1
      self.writer.add_scalar('surprise', surprise, self.metric_update_steps)
    else:
      self.cumulated_state.append(state[0])
      surprise, features = self.metric(state.to(self.device))
    surprise = surprise.cpu().data.numpy()
    features = features.flatten().cpu().data.numpy()

    agent['bs'] = np.array([[obs[0][0], obs[0][1]]])
    agent['features'] = [features, state.cpu().data.numpy()]
    agent['surprise'] = surprise
    agent['reward'] = cumulated_reward

  def update_archive_feat(self):
    """
    This function is used to update the position of the archive elements in the feature space (given that is changing
    while the AE learns)
    :return:
    """
    # TODO this one should be done batch wise! Instead of a for loop, put all the states in a tensor and do a single pass
    for agent in self.archive:
      state = torch.Tensor(agent['features'][1]).to(self.device)
      _, feature = self.metric(state)
      agent['features'][0] = feature.flatten().cpu().data.numpy()

  def update_metric(self):
    """
    This function uses the cumulated state to update the metrics parameters and then empties the cumulated_state
    :return:
    """
    self.cumulated_state = torch.stack(self.cumulated_state).to(self.device)
    cum_surprise = []
    # Split the batch in 3 minibatches to have better learning
    mini_batches = utils.split_array(self.cumulated_state, wanted_parts=3)
    for data in mini_batches:
      cum_surprise.append(self.metric.training_step(data))
      self.metric_update_steps += 1
      self.writer.add_scalar('novelty', cum_surprise[-1][0], self.metric_update_steps)

    features = torch.cat([cs[1] for cs in cum_surprise])
    self.cumulated_state = []
    return cum_surprise[-1][0], features

  def train(self, steps=10000):
    """
    This function trains the agents and the RND
    :param steps: number of update steps (or generations)
    :return:
    """
    self.elapsed_gen = 0
    for self.elapsed_gen in range(steps):
      cs = 0
      max_rew = -np.inf
      for a in self.population:
        self.evaluate_agent(a)
        cs += a['surprise']
        if max_rew < a['reward']:
          max_rew = a['reward']

      if not self.params.optimizer_type == 'Surprise':
        self.update_archive_feat()
      self.opt.step()

      # Has to be done after the archive features have been updated cause pop and archive need to have features from the same update step.
      if not self.metric_update_single_agent:
        self.update_metric()

      self.writer.add_scalar('Archive_size', self.archive.size, self.elapsed_gen)
      self.writer.add_scalar('Avg_generation_novelty', cs/self.pop_size)

      if self.elapsed_gen % 1 == 0:
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
    save_subf = os.path.join(self.save_path, 'models')
    print('Saving...')
    if not os.path.exists(save_subf):
      try:
        os.makedirs(os.path.abspath(save_subf))
      except:
        print('Cannot create save folder.')
    self.population.save_pop(save_subf, 'pop')
    self.archive.save_pop(save_subf, 'archive')
    self.metric.save(save_subf)

    self.writer.export_scalars_to_json(os.path.join(self.save_path, "scalars_log.json"))
    self.writer.close()

    print('Done')
