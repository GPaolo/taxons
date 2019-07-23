import numpy as np
from core.metrics import rnd, ae
from core.qd import population, agents
from core.utils import utils
import torch
import os
import json
import gc


class RndQD(object):

  # ---------------------------------------------------
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

    print("Seed {} - Using device: {}".format(self.params.seed, self.device))

    if self.params.metric == 'AE':
      self.metric = ae.AutoEncoder(device=self.device,
                                   learning_rate=self.params.learning_rate,
                                   lr_scale=self.params.lr_scale_fact,
                                   encoding_shape=self.params.feature_size)
    elif self.params.metric == 'FFAE':
      self.metric = ae.FFAE(device=self.device,
                            learning_rate=self.params.learning_rate,
                            lr_scale=self.params.lr_scale_fact,
                            encoding_shape=self.params.feature_size)
    elif self.params.metric == 'BVAE':
      self.metric = ae.BVAE(device=self.device, learning_rate=self.params.learning_rate, encoding_shape=self.params.feature_size)
    else:
      self.metric = rnd.RND(device=self.device, learning_rate=self.params.learning_rate, encoding_shape=self.params.feature_size)

    self.opt = self.params.optimizer(self.population, archive=self.archive, mutation_rate=self.params.mutation_rate, metric_update_interval=self.params.update_interval)

    self.END = False
    self.elapsed_gen = 0
  # ---------------------------------------------------

  # ---------------------------------------------------
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
      obs = utils.obs_formatting(self.params.env_tag, obs, reward, done, info)
      t += 1
      cumulated_reward += reward

      if 'Ant' in self.params.env_tag:
        CoM = np.array([self.env.env.data.qpos[:2]])
        if t >= self.params.max_episode_len or np.any(np.abs(CoM) >= np.array([3, 3])):
          done = True
    state = self.env.render(mode='rgb_array')
    state = state/np.max((np.max(state), 1))

    if 'Ant' in self.params.env_tag:
      agent['bs'] =  np.array([self.env.env.data.qpos[:2]]) # xy position of CoM of the robot
    else:
      agent['bs'] = np.array([[obs[0][0], obs[0][1]]])
    agent['reward'] = cumulated_reward
    return state, None, cumulated_reward # TODO check why there is a None here
  # ---------------------------------------------------

  # ---------------------------------------------------
  def update_agents(self, states):
    surprise, features, _ = self.metric(states.to(self.device))
    surprise = surprise.cpu().data.numpy() # Has dimension [pop_size]
    features = features.cpu().data.numpy()

    for agent, state, feat, surpr in zip(self.population, states, features, surprise):
      agent['features'] = [feat, state.cpu().data.numpy()]
      agent['surprise'] = surpr
    return surprise
  # ---------------------------------------------------

  # ---------------------------------------------------
  def update_archive_feat(self):
    """
    This function is used to update the position of the archive elements in the feature space (given that is changing
    while the AE learns)
    :return:
    """
    if not len(self.archive) == 0:
      feats = self.archive['features'].values
      state = torch.Tensor([f[1] for f in feats]).to(self.device)
      surprise, feature, _ = self.metric(state)
      surprise = surprise.cpu().data.numpy()  # Has dimension [pop_size]

      for agent, feat in zip(self.archive, feature):
        agent['features'][0] = feat.flatten().cpu().data.numpy()
      self.archive.pop['surprise'] = surprise
  # ---------------------------------------------------

  # ---------------------------------------------------
  def update_metric(self, states, old_states=None):
    """
    This function uses the cumulated state to update the metrics parameters and then empties the cumulated_state
    :return:
    """
    # Take archive data
    if not len(self.archive) == 0 and self.params.train_on_archive:
      feats = self.archive['features'].values
      archi_state = torch.Tensor([f[1] for f in feats]).to(self.device)
      total_state = torch.cat((states.to(self.device), archi_state), 0)
    else:
      total_state = states.to(self.device)
    # old_states = old_states.to(self.device)
    # Split the batch in 3 minibatches to have better learning
    mini_batches = utils.split_array(total_state, batch_size=128)
    # old_mini_batches = utils.split_array(old_states, batch_size=64)
    # for data, old_data in zip(mini_batches, old_mini_batches):
    for data in mini_batches:
      loss, f, _ = self.metric.training_step(data)
      self.metric_update_steps += 1
    # print("Loss at {}: {}".format(self.metric_update_steps, loss))
    return f
  # ---------------------------------------------------

  # ---------------------------------------------------
  def train(self, steps=10000):
    """
    This function trains the agents and the RND
    :param steps: number of update steps (or generations)
    :return:
    """
    inputs = None
    if 'Ant' in self.params.env_tag: # Need it otherwise cannot init OpenGL
      self.env.render()
    for self.elapsed_gen in range(steps):
      states = []
      for agent in self.population:
        state, _, _ = self.evaluate_agent(agent)
        states.append(state)
      states = np.stack(states)# - self.running_avg # Center data for training
      states = self.metric.subsample(torch.Tensor(states).permute(0, 3, 1, 2))
      if self.params.update_metric:
        if inputs is None:
          inputs = states.clone()
        else:
          inputs = torch.cat((inputs, states), 0)

      avg_gen_surprise = np.mean(self.update_agents(states))
      max_rew = np.max(self.population['reward'].values)

      if self.params.update_metric and not self.params.optimizer_type == 'Surprise':
        self.update_archive_feat()
      self.opt.step()

      # Has to be done after the archive features have been updated cause pop and archive need to have features from the same update step.
      if self.params.update_metric and self.elapsed_gen % self.params.update_interval == 0 and self.elapsed_gen > 0:
        for epoch in range(5):
          f = self.update_metric(inputs)
          print(f[0].cpu().data)
        del inputs
        inputs = None

      # if hasattr(self.metric, 'lr_scheduler') and self.elapsed_gen % 100 == 0 and self.elapsed_gen > 0:
      #   self.metric.lr_scheduler.step()

      torch.cuda.empty_cache()
      if self.elapsed_gen % 10 == 0:
        gc.collect()
        print('Seed {} - Generation {}'.format(self.params.seed, self.elapsed_gen))
        if self.archive is not None:
          print('Seed {} - Archive size {}'.format(self.params.seed, self.archive.size))
        print('Seed {} - Average generation surprise {}'.format(self.params.seed, avg_gen_surprise))
        print('Seed {} - Max reward {}'.format(self.params.seed, max_rew))
        print('Saving checkpoint...')
        self.save(ckpt=True)
        print("Done")
        print()

      if self.archive is not None:
        bs_points = np.concatenate(self.archive['bs'].values)
      else:
        bs_points = np.concatenate([a['bs'] for a in self.population if a['bs'] is not None])
      if 'Ant' in self.params.env_tag:
        limit = 3.5
      else:
        limit = 1.35
      coverage = utils.show(bs_points, filepath=self.save_path, info={'gen':self.elapsed_gen, 'seed':self.params.seed}, limit=limit)

      self.logs['Generation'].append(str(self.elapsed_gen))
      self.logs['Avg gen surprise'].append(str(avg_gen_surprise))
      self.logs['Max reward'].append(str(max_rew))
      self.logs['Archive size'].append(str(self.archive.size))
      self.logs['Coverage'].append(str(coverage))
      if self.END:
        print('Seed {} - Quitting.'.format(self.params.seed))
        break
    gc.collect()
  # ---------------------------------------------------

  # ---------------------------------------------------
  def save(self, ckpt=False):
    if ckpt:
      folder = 'models/ckpt'
    else:
      folder = 'models'
    save_subf = os.path.join(self.save_path, folder)
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
  # ---------------------------------------------------
