import numpy as np
from core.metrics import rnd, ae
from core.qd import population, agents
from core.utils import utils
import gym, torch
import gym_billiard
import os, threading, sys, traceback
import matplotlib
import json
import gc


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

    self.metric_update_steps = 0
    self.metric_update_single_agent = self.params.per_agent_update
    self.logs = {'Generation':[], 'Avg gen surprise':[], 'Max reward':[], 'Archive size':[], 'Coverage':[]}

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
    self.elapsed_gen = 0

    # self.thread = threading.Thread(target=self._control_interface)
    # self.thread.daemon = True
    # self.thread.start()

  # Need these two functions to remove pool from the dict
  # def __getstate__(self):
  #   self_dict = self.__dict__.copy()
  #   del self_dict['pool']
  #   del self_dict['thread']
  #   return self_dict
  #
  # def __setstate__(self, state):
  #   self.__dict__.update(state)
  #
  # def _control_interface(self):
  #   print('If you want to show the progress, press s.')
  #   print('If you want to stop training, press q.')
  #   matplotlib.use('agg')
  #   while True:
  #     try:
  #       action = input(' ')
  #       if action == 's':
  #         try:
  #           if self.archive is not None:
  #             bs_points = np.concatenate(self.archive['bs'].values)
  #           else:
  #             bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
  #           utils.show(bs_points, filepath=self.save_path)
  #         except BaseException as e:
  #           ex_type, ex_value, ex_traceback = sys.exc_info()
  #           trace_back = traceback.extract_tb(ex_traceback)
  #           stack_trace = list()
  #           for trace in trace_back:
  #             stack_trace.append(
  #               "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
  #           print('Cannot show progress due to {}: {}'.format(ex_type.__name__, ex_value))
  #           print(stack_trace[0])
  #       elif action == 'q':
  #         print('Quitting training...')
  #         self.END = True
  #         break
  #     except KeyboardInterrupt:
  #       print('BYE')
  #       break

  def evaluate_agent(self, agent):
    """
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    """
    done = False
    cumulated_reward = 0

    obs = utils.obs_formatting(self.params.env_tag, self.env.reset())
    t = 0
    while not done:
      agent_input = t
      action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input))
      obs, reward, done, info = self.env.step(action)
      obs = utils.obs_formatting(self.params.env_tag, obs)
      t += 1
      cumulated_reward += reward

    state = self.env.render(mode='rgb_array')
    agent['bs'] = np.array([[obs[0][0], obs[0][1]]])
    agent['reward'] = cumulated_reward
    return state

  def update_agents(self, states):
    states = self.metric.subsample(torch.Tensor(states).permute(0, 3, 1, 2))
    if self.params.update_metric:
      self.cumulated_state = states
    surprise, features = self.metric(states.to(self.device))
    surprise = surprise.cpu().data.numpy() # Has dimension [pop_size]
    features = features.cpu().data.numpy()

    for agent, state, feat, surpr in zip(self.population, states, features, surprise):
      agent['features'] = [feat, state.cpu().data.numpy()]
      agent['surprise'] = surpr
    return surprise

  def update_archive_feat(self):
    """
    This function is used to update the position of the archive elements in the feature space (given that is changing
    while the AE learns)
    :return:
    """
    if not len(self.archive) == 0:
      feats = self.archive['features'].values
      state = torch.Tensor([f[1] for f in feats]).to(self.device)
      _, feature = self.metric(state)

      for agent, feat in zip(self.archive, feature):
        agent['features'][0] = feat.flatten().cpu().data.numpy()

  def update_metric(self):
    """
    This function uses the cumulated state to update the metrics parameters and then empties the cumulated_state
    :return:
    """
    # Split the batch in 3 minibatches to have better learning
    mini_batches = utils.split_array(self.cumulated_state.to(self.device), wanted_parts=3)
    for data in mini_batches:
      self.metric.training_step(data)
      self.metric_update_steps += 1

  def train(self, steps=10000):
    """
    This function trains the agents and the RND
    :param steps: number of update steps (or generations)
    :return:
    """
    for self.elapsed_gen in range(steps):
      states = []
      for agent in self.population:
        states.append(self.evaluate_agent(agent))
      states = np.stack(states)
      avg_gen_surprise = np.mean(self.update_agents(states))
      max_rew = np.max(self.population['reward'].values)

      if self.params.update_metric and not self.params.optimizer_type == 'Surprise':
        self.update_archive_feat()
      self.opt.step()

      # Has to be done after the archive features have been updated cause pop and archive need to have features from the same update step.
      if self.params.update_metric and not self.metric_update_single_agent:
        self.update_metric()

      if self.elapsed_gen % 10 == 0:
        gc.collect()
        print('Seed {} - Generation {}'.format(self.params.seed, self.elapsed_gen))
        if self.archive is not None:
          print('Seed {} - Archive size {}'.format(self.params.seed, self.archive.size))
        print('Seed {} - Average generation surprise {}'.format(self.params.seed, avg_gen_surprise))
        print('Seed {} - Max reward {}'.format(self.params.seed, max_rew))
        print()

      if self.archive is not None:
        bs_points = np.concatenate(self.archive['bs'].values)
      else:
        bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
      coverage = utils.show(bs_points, filepath=self.save_path, info={'gen':self.elapsed_gen})

      self.logs['Generation'].append(str(self.elapsed_gen))
      self.logs['Avg gen surprise'].append(str(avg_gen_surprise))
      self.logs['Max reward'].append(str(max_rew))
      self.logs['Archive size'].append(str(self.archive.size))
      self.logs['Coverage'].append(str(coverage))
      if self.END:
        print('Seed {} - Quitting.'.format(self.params.seed))
        break

  def save(self):
    save_subf = os.path.join(self.save_path, 'models')
    print('Seed {} - Saving...'.format(self.params.seed))
    if not os.path.exists(save_subf):
      try:
        os.makedirs(os.path.abspath(save_subf))
      except:
        print('Seed {} - Cannot create save folder.'.format(self.params.seeds))
    self.population.save_pop(save_subf, 'pop')
    self.archive.save_pop(save_subf, 'archive')
    self.metric.save(save_subf)

    with open(os.path.join(self.save_path, 'logs.json'), 'w') as f:
      json.dump(self.logs, f, indent=4)
    print('Seed {} - Done'.format(self.params.seed))
