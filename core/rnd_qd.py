import numpy as np
from core.metrics import rnd, ae
from core.qd import population, agents
from core.utils import utils
import gym, torch
import gym_billiard
import os, threading, sys, traceback
import matplotlib
import multiprocessing.dummy as mp
import simplejson as json


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
    self.logs = {'Generation':[], 'Avg gen surprise':[], 'Max reward':[], 'Archive size':[], }

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

    print("Using device: {}".format(self.device))

    if self.params.metric == 'AE':
      self.metric = ae.ConvAutoEncoder(device=self.device, learning_rate=self.params.learning_rate, encoding_shape=self.params.feature_size)
    else:
      self.metric = rnd.RND(device=self.device, learning_rate=self.params.learning_rate, encoding_shape=self.params.feature_size)

    self.opt = self.params.optimizer(self.population, archive=self.archive)
    self.cumulated_state = []

    self.END = False
    self.thread = threading.Thread(target=self._control_interface)
    self.thread.daemon = True
    self.thread.start()
    if self.params.parallel:

      self.pool = mp.Pool()

  # Need these two functions to remove pool from the dict
  def __getstate__(self):
    self_dict = self.__dict__.copy()
    del self_dict['pool']
    return self_dict

  def __setstate__(self, state):
    self.__dict__.update(state)

  def _control_interface(self):
    print('If you want to show the progress, press s.')
    print('If you want to stop training, press q.')
    matplotlib.use('agg')
    while True:
      try:
        action = input(' ')
        if action == 's':
          try:
            if self.archive is not None:
              bs_points = np.concatenate(self.archive['bs'].values)
            else:
              bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
            if self.params.env_tag == 'Ant-v2':
              limit = 10
            else:
              limit = 1.35
            utils.show(bs_points, filepath=self.save_path, limit=limit)
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
          break
      except KeyboardInterrupt:
        print('BYE')
        break

  def evaluate_agent(self, agent_env):
    """
    This function evaluates the agent in the environment. This function should be run in parallel
    :param agent: agent to evaluate
    :return:
    """
    done = False
    cumulated_reward = 0

    obs = utils.obs_formatting(self.params.env_tag, agent_env[1].reset())
    t = 0
    while not done:
      if self.agent_name == 'Neural':
        agent_input = obs
      elif self.agent_name == 'DMP':
        agent_input = t

      action = utils.action_formatting(self.params.env_tag, agent_env[0]['agent'](agent_input))
      obs, reward, done, info = agent_env[1].step(action)
      obs = utils.obs_formatting(self.params.env_tag, obs)
      t += 1
      cumulated_reward += reward

    state = agent_env[1].render(mode='rgb_array')
    if self.params.env_tag == 'Ant-v2':
      agent_env[0]['bs'] = np.array([agent_env[1].env.data.qpos[:2]]) # xy position of CoM of the robot
    else:
      agent_env[0]['bs'] = np.array([[obs[0][0], obs[0][1]]])
    # agent['features'] = [features, state.cpu().data.numpy()]
    # agent['surprise'] = surprise
    agent_env[0]['reward'] = cumulated_reward
    return state

  def update_agents(self, states):
    try: # FFAE does not have any subsampling
      states = self.metric.subsample(torch.Tensor(states).permute(0, 3, 1, 2))
    except AttributeError:
      states = torch.Tensor(states)

    # if self.metric_update_single_agent and self.params.update_metric:
    #   surprise, features = self.metric.training_step(state.to(self.device))  # Input Dimensions need to be [1, input_dim]
    #   self.metric_update_steps += 1
    # else:
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


    # for agent in self.archive:
    #   state = torch.Tensor(agent['features'][1]).to(self.device)
    #   _, feature = self.metric(state)
    #   agent['features'][0] = feature.flatten().cpu().data.numpy()

  def update_metric(self):
    """
    This function uses the cumulated state to update the metrics parameters and then empties the cumulated_state
    :return:
    """
    # self.cumulated_state = torch.stack(self.cumulated_state).to(self.device)
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
    self.elapsed_gen = 0
    for self.elapsed_gen in range(steps):
      if self.params.parallel:
        states = self.pool.map(self.evaluate_agent, zip(self.population, self.env))
      else:
        states = []
        for agent in self.population:
          states.append(self.evaluate_agent((agent, self.env[0])))
      states = np.stack(states)
      avg_gen_surprise = np.mean(self.update_agents(states))
      max_rew = np.max(self.population['reward'].values)

      if self.params.update_metric and not self.params.optimizer_type == 'Surprise':
        self.update_archive_feat()
      self.opt.step()

      # Has to be done after the archive features have been updated cause pop and archive need to have features from the same update step.
      if self.params.update_metric and not self.metric_update_single_agent:
        self.update_metric()

      if self.elapsed_gen % 1 == 0:
        print('Generation {}'.format(self.elapsed_gen))
        if self.archive is not None:
          print('Archive size {}'.format(self.archive.size))
        print('Average generation surprise {}'.format(avg_gen_surprise))
        print('Max reward {}'.format(max_rew))
        print()

      self.logs['Generation'].append(str(self.elapsed_gen))
      self.logs['Avg gen surprise'].append(str(avg_gen_surprise))
      self.logs['Max reward'].append(str(max_rew))
      self.logs['Archive size'].append(str(self.archive.size))

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

    with open(os.path.join(self.save_path, 'logs.json'), 'w') as f:
      json.dump(self.logs, f, indent=4)
    print('Done')
