# Created by giuseppe
# Date: 28/03/19

from scripts import parameters
import gym, torch
import gym_billiard, gym_fastsim, pybulletgym
from gym.wrappers import Monitor
import numpy as np
from core.metrics import ae, rnd
from core.evolution import population, agents
from core.utils import utils
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pkl
import progressbar
import gc
import pyfastsim as fs




class Eval(object):

  # -----------------------------------------------
  def __init__(self, exp_folder=None, reeval_bs=False, targets=100, render_test=False):
    assert os.path.exists(exp_folder), 'Experiment folder {} does not exist'.format(exp_folder)
    self.folder = exp_folder
    self.params = None
    self.reeval_bs = reeval_bs
    self.render_test = render_test

    # Get all the seeds
    self.seeds = list(os.walk(self.folder))[0][1][:1]

    if 'Billiard' in self.folder:
      self.env_tag = 'Billiard-v0'
    elif 'Ant' in self.folder:
      self.env_tag = 'AntMuJoCoEnv-v0'
    elif 'Maze' in self.folder:
      self.env_tag = 'FastsimSimpleNavigation-v0'

    self.env = gym.make(self.env_tag)
    self.env.reset()

    self.target_images, self.target_poses = self.generate_targets(targets)
  # -----------------------------------------------

  # -----------------------------------------------
  def load_params(self, path):
    print('Loading parameters...')
    self.params = parameters.Params()
    self.params.load(os.path.join(path, 'params.json'))
    assert self.env_tag == self.params.env_tag, 'Env tag from folder different from parameters env tag: {} - {}'.format(self.env_tag, self.params.env_tag)
  # -----------------------------------------------

  # -----------------------------------------------
  def generate_targets(self, targets, plot_samples=False):
    """
    This function generates the target examples on which the agents are going to be tested
    :return: target_images, target_poses
    """
    print('Generating targets...')
    target_images = []
    target_poses = []
    if "Billiard" in self.env_tag:
      self.env.env.params.RANDOM_BALL_INIT_POSE = True
    #elif "Ant" in self.env_tag:
    #  self.env.render()

    with progressbar.ProgressBar(max_value=targets) as bar:
      for k in range(targets):  # Generate target datapoints
        obs = self.env.reset()

        if "Ant" in self.env_tag:
          for step in range(300):
            self.env.step(self.env.action_space.sample())
            CoM = np.array([self.env.env.data.qpos[:2]])
            t_pose = CoM
            if np.any(np.abs(CoM) >= np.array([3, 3])):
              break
        elif 'Billiard' in self.env_tag:
          t_pose = obs[0]

        tmp = self.env.render(mode='rgb_array')
        target_images.append(tmp)
        target_poses.append(t_pose)

        bar.update(k)

    target_images = np.stack(target_images)
    target_poses = np.stack(target_poses)

    if plot_samples:
      fig, ax = plt.subplots(4, 5)
      k = 0
      for i in range(4):
        for j in range(5):
          a = target_images[k]
          ax[i, j].imshow(a)
          ax[i, j].set_title(k)
          ax[i, j].set_ylabel(target_poses[k][1])
          ax[i, j].set_xlabel(target_poses[k][0])
          k += 1
      plt.subplots_adjust(left=0.02, right=.99, top=0.95, bottom=0.02, wspace=0.4)
      plt.show()

    if "Billiard" in self.env_tag:
      self.env.env.params.RANDOM_BALL_INIT_POSE = False
    print('Done.')

    return target_images, np.squeeze(target_poses)
  # -----------------------------------------------

  # -----------------------------------------------
  def _get_target_bs_point(self, image=None, pose=None):
    if 'AE' in self.folder:
      goal = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
      goal = goal / torch.max(torch.Tensor(np.array([torch.max(goal).cpu().data, 1])))  # Normalize in [0,1]
      surprise, bs_point, reconstr = self.selector(goal)
      bs_point = bs_point.flatten().cpu().data.numpy()
    elif 'NS' in self.folder:
      bs_point = pose
    elif 'RBD' in self.folder:
      bs_point = np.random.random(self.params.feature_size)
    else:
      raise ValueError('This experiment cannot be tested. Only ones are AE, NS or RBD')

    return bs_point
  # -----------------------------------------------

  # -----------------------------------------------
  def load_metric(self, load_path):
    print('Loading metric...')
    if self.params.gpu:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device('cpu')

    if self.params.metric == 'AE':
      self.selector = ae.ConvAE(device=self.device, encoding_shape=self.params.feature_size)
      self.selector.load(os.path.join(load_path, 'models/ckpt_ae.pth'))
    elif self.params.metric == 'RND':
      self.selector = rnd.RND(self.params.feature_size)
      self.selector.load(os.path.join(load_path, 'models/ckpt_rnd.pth'))
    else:
      raise ValueError('Wrong metric selected: {}'.format(self.params.metric))
  # -----------------------------------------------

  # -----------------------------------------------
  def load_archive(self, load_path):
    print('Loading agents...')
    if self.params.qd_agent == 'Neural':
      agent_type = agents.FFNeuralAgent
    elif self.params.qd_agent == 'DMP':
      agent_type = agents.DMPAgent
    else:
      raise ValueError('Wrong agent type selected: {}'.format(self.params.qd_agent))

    self.pop = population.Population(agent=agent_type, pop_size=0, shapes=self.params.agent_shapes)
    self.pop.load_pop(os.path.join(load_path, 'models/qd_archive.pkl'))
  # -----------------------------------------------

  # -----------------------------------------------
  def evaluate_archive_bs(self):
    """
    This one calculates the pop bs points. Might not need it cause they are saved, with the pop already
    """
    for i, agent in enumerate(self.pop):
      if i % 50 == 0:
        print('Evaluating agent {}'.format(i))

      state, _ = self._test_agent(agent)
      state = state / np.max((np.max(state), 1))
      state = self.selector.subsample(torch.Tensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device))
      surprise, bs_point, y = self.selector(state)
      bs_point = bs_point.flatten().cpu().data.numpy()
      agent['features'] = [bs_point]
  # -----------------------------------------------

  # -----------------------------------------------
  def _get_closest_agent(self, bs_point):
    bs_space = np.stack([a[0] for a in self.pop['features'].values])
    # Get distances
    diff = np.atleast_2d(bs_space - bs_point)
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    # Get agent with smallest distance in BS space
    closest_agent = np.argmin(dists)
    selected = self.pop[closest_agent]
    # print("Selected agent {}".format(closest_agent))
    return selected
  # -----------------------------------------------

  # -----------------------------------------------
  def _test_agent(self, agent):
    """
    Tests agent in the environment
    :param agent:
    :return: final state image and final (x,y) pose
    """
    done = False
    ts = 0
    obs = utils.obs_formatting(self.env_tag, self.env.reset())

    while not done:
      if self.render_test:
        self.env.render()
      agent_input = ts
      action = utils.action_formatting(self.env_tag, agent['agent'](agent_input))# TODO /self.params.max_episode_len))

      obs, reward, done, info = self.env.step(action)
      obs = utils.obs_formatting(self.env_tag, obs, reward, done, info)
      ts += 1

      if ts >= self.params.max_episode_len:
        done = True

      if 'Ant' in self.env_tag:
        CoM = np.array(self.env.env.data.qpos[:2])
        f_pose = CoM
        if np.any(np.abs(CoM) >= np.array([3, 3])):
          done = True
      elif 'Billiard' in self.env_tag:
        f_pose = obs[0][:2]

    state = self.env.render(mode='rgb_array')
    state = state / np.max((np.max(state), 1))
    gc.collect()
    return state, f_pose
  # -----------------------------------------------

  # -----------------------------------------------
  def run_test(self):
    """
    This is the main function that calls the others. It loads all the needed things and runs the test for each seed
    :return: displacement errors for each seed
    """
    errors = {}
    for seed in self.seeds:
      print('')
      print('Evaluating seed {}...'.format(seed))
      load_path = os.path.join(self.folder, seed)

      self.load_params(load_path)
      if 'AE' in self.folder:
        self.load_metric(load_path)
      self.load_archive(load_path)

      self.env.seed(int(seed))
      if self.pop[0]['features'] is None or self.reeval_bs:
        self.evaluate_archive_bs()

      final_state, final_pose_error = self.test_pop()
      errors[seed] = np.copy(final_pose_error)

      to_save = errors.copy()
      to_save['targets'] = self.target_poses
      with open(os.path.join(self.folder, 'errors.pkl'), 'wb') as f:
        pkl.dump(to_save, f)

    return errors
  # -----------------------------------------------

  # -----------------------------------------------
  def test_pop(self):
    """
    This one tests the pop related to the seed
    :return: array of final image states and array of final positional errors
    """
    print('Starting pop testing...')
    final_pose = []
    final_state = []

    with progressbar.ProgressBar(max_value=len(self.target_poses)) as bar:
      for target_idx in range(len(self.target_images)):
        # Get BS point
        bs_point = self._get_target_bs_point(self.target_images[target_idx], self.target_poses[target_idx])

        selected = self._get_closest_agent(bs_point)
        state, f_pose = self._test_agent(selected)

        if self.render_test:
          fig, ax = plt.subplots(2)
          ax[0].imshow(state)
          ax[0].set_title('Final State')
          ax[1].imshow(self.target_images[target_idx])
          ax[1].set_title('Target state')
          # ax[2].imshow(reconstr.permute(0,2,3,1)[0].cpu().data)
          # ax[2].set_title('Reconstructed')
          plt.show()

        final_pose.append(f_pose)
        # final_state.append(state)

        final_distance = np.sqrt(np.sum((self.target_poses[target_idx] - f_pose) ** 2))
        bar.update(target_idx)
        # print('Positional error: {}'.format(final_distance))

    gc.collect()
    # final_state = np.stack(final_state)
    final_pose = np.squeeze(np.stack(final_pose))
    final_pose_error = np.sqrt(np.sum((self.target_poses - final_pose) ** 2, axis=1))
    print('Done')
    return final_state, final_pose_error
  # -----------------------------------------------

  # -----------------------------------------------
  def plot_errors(self, errors):
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(60, 10))
    size = (50, 50)
    heatmap = np.zeros(size)
    counts = np.ones_like(heatmap) # This one is used to do the avg for each position

    if 'Ant' in self.folder:
      data_range = [-3.5, 3.5]
    elif 'Billiard' in self.folder:
      data_range = [-1.3, 1.3]

    points = (self.target_poses-data_range[0])*size/(2*data_range[1])
    points = points.astype(int)

    for seed in errors:
      for point, error in zip(points, errors[seed]):
        heatmap[point[0], point[1]] += error #TODO Might have to come up with a beter way to deal with errors in the same positions
        counts[point[0], point[1]] += 1

    heatmap = heatmap/counts

    fig, axes = plt.subplots(nrows=1, ncols=1)

    im = axes.imshow(heatmap, cmap=cm.jet)#, interpolation='bessel')
    cb = fig.colorbar(im, ax=axes)
    cb.set_label('mean value')
    plt.show()
  # -----------------------------------------------

# -----------------------------------------------




if __name__ == "__main__":
  # evaluator = Eval(exp_folder='/home/giuseppe/src/rnd_qd/experiments/Billiard_AE_Mixed/11', targets=30, render_test=True)


  # errors = evaluator.run_test()
  # evaluator.plot_errors(errors)

  # print("")
  # for key in errors:
  #   avg = np.mean(errors[key])
  #   std = np.std(errors[key])
  #   print("Seed {}: Mean {} - Std {}.".format(key, avg, std))




  # Parameters
  # -----------------------------------------------
  seed = 42
  load_path = '/home/giuseppe/src/rnd_qd/experiments/Maze_AE_Mixed/{}'.format(seed)

  params = parameters.Params()
  params.load(os.path.join(load_path, 'params.json'))

  env = gym.make(params.env_tag)
  # -----------------------------------------------

  # Possible targets
  # -----------------------------------------------
  # x = []
  # target_poses = []
  # if "Billiard" in params.env_tag:
  #   env.env.params.RANDOM_BALL_INIT_POSE = True
  # elif "Ant" in params.env_tag:
  #   env.render()
  #
  # for k in range(50): # Generate 50 target datapoints
  #   obs = env.reset()
  #
  #   if "Ant" in params.env_tag:
  #     for step in range(300):
  #       env.step(env.action_space.sample())
  #       CoM = np.array([env.env.data.qpos[:2]])
  #       t_pose = CoM
  #       if np.any(np.abs(CoM) >= np.array([3, 3])):
  #         break
  #   elif 'Billiard' in params.env_tag:
  #     t_pose = obs[0]
  #
  #   tmp = env.render(mode='rgb_array')
  #   x.append(tmp)
  #   target_poses.append(t_pose)
  #
  # x = np.stack(x)
  # target_poses = np.stack(target_poses)
  #
  # fig, ax = plt.subplots(4, 5)
  # k = 0
  # for i in range(4):
  #   for j in range(5):
  #     a = x[k]
  #     ax[i, j].imshow(a)
  #     ax[i, j].set_title(k)
  #     ax[i, j].set_ylabel(target_poses[k][1])
  #     ax[i, j].set_xlabel(target_poses[k][0])
  #     k += 1
  # plt.subplots_adjust(left=0.02, right=.99, top=0.95, bottom=0.02, wspace=0.4)
  # plt.show()
  #
  # if "Billiard" in params.env_tag:
  #   env.env.params.RANDOM_BALL_INIT_POSE = False
  # -----------------------------------------------
  target_pose = [450, 500]
  # Generate target image
  if "Billiard" in params.env_tag:
    env.reset()
    obs = env.reset(desired_ball_pose=target_pose)
    target_image = env.render(mode='rgb_array')
    plt.figure()
    plt.imshow(target_image)
    plt.show()
  elif "Ant" in params.env_tag:
    env.render()
    env.reset()
    qpos = np.zeros(15)
    qvel = np.zeros(14)
    qpos[:2] = np.array(target_pose)
    env.env.set_state(qpos, qvel)
    target_image = env.render(mode='rgb_array')
    plt.figure()
    plt.imshow(target_image)
    plt.show()
    # env.env.robot_body.reset_position([target_pose[0], target_pose[1], 0.5])
    # target_image = env.render(mode='rgb_array', top_bottom=True)
    #plt.figure()
    #plt.imshow(target_image)
    #plt.show()
  elif 'Fastsim' in params.env_tag:
    obs = env.reset()
    pose = target_pose + env.initPos[-1:]
    p = fs.Posture(*pose)
    env.robot.set_pos(p)
    env.enable_display()
    target_image = env.render(mode='rgb_array', target=False)
    plt.figure()
    plt.imshow(target_image)
    plt.show()


  # Load metric
  # -----------------------------------------------
  print('Loading metric...')
  if params.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device('cpu')

  if params.metric == 'AE':
    selector = ae.ConvAE(device=device, encoding_shape=params.feature_size)
    selector.load(os.path.join(load_path, 'models/ckpt_ae.pth'))
  elif params.metric == 'RND':
    selector = rnd.RND(params.feature_size)
    selector.load(os.path.join(load_path, 'models/ckpt_rnd.pth'))
  else:
    raise ValueError('Wrong metric selected: {}'.format(params.metric))
  # -----------------------------------------------

  # Load archive
  # -----------------------------------------------
  print('Loading agents...')
  if params.qd_agent == 'Neural':
    agent_type = agents.FFNeuralAgent
  elif params.qd_agent == 'DMP':
    agent_type = agents.DMPAgent
  else:
    raise ValueError('Wrong agent type selected: {}'.format(params.qd_agent))

  pop = population.Population(agent=agent_type, pop_size=0, shapes=params.agent_shapes)
  pop.load_pop(os.path.join(load_path, 'models/qd_archive.pkl'))
  # -----------------------------------------------

  # Evaluate archive agents BS points
  # -----------------------------------------------
  if pop[0]['features'] is None:
    for i, agent in enumerate(pop):
      if i % 50 == 0:
        print('Evaluating agent {}'.format(i))
      done = False
      obs = env.reset()
      t = 0
      while not done:
        agent_input = t
        action = utils.action_formatting(params.env_tag, agent['agent'](agent_input))

        obs, reward, done, info = env.step(action)
        t += 1

        if t >= params.max_episode_len:
          done = True

        if "Ant" in params.env_tag:
          CoM = np.array([env.env.data.qpos[:2]])
          if np.any(np.abs(CoM) >= np.array([3, 3])):
            done = True

      state = env.render(mode='rgb_array')
      state = state/np.max((np.max(state), 1))
      state = selector.subsample(torch.Tensor(state).permute(2, 0, 1).unsqueeze(0).to(device))
      surprise, bs_point, y = selector(state)
      bs_point = bs_point.flatten().cpu().data.numpy()
      agent['features'] = [bs_point]
  # -----------------------------------------------
  #
  # Automatic testing
  # -----------------------------------------------
  print('Starting testing...')
  # final_pose = []
  # final_state = []

  # for target_idx in range(len(x)):
  #   print('Testing target {}'.format(target_idx))

  # Get target image BS point
  goal = torch.Tensor(np.ascontiguousarray(target_image)).permute(2, 0, 1).unsqueeze(0).to(device)
  goal = goal/torch.max(torch.Tensor(np.array([torch.max(goal).cpu().data, 1]))) #Normalize in [0,1]
  surprise, bs_point, reconstr = selector(goal)
  bs_point = bs_point.flatten().cpu().data.numpy()
  print('Target surprise {}'.format(surprise.cpu().data))
  print('Target bs point {}'.format(bs_point))
  print('')
  plt.figure()
  plt.imshow(reconstr.permute(0, 2, 3, 1)[0].cpu().data)
  plt.show()
  #
  # Get closest agent
  # -----------------------------------------------
  bs_space = np.stack([a[0] for a in pop['features'].values])
  # Get distances
  diff = np.atleast_2d(bs_space - bs_point)
  dists = np.sqrt(np.sum(diff * diff, axis=1))
  # Get agent with smallest distance in BS space
  closest_agent = np.argmin(dists)
  selected = pop[closest_agent]
  print("Selected agent {} - Distance {}".format(closest_agent, dists[closest_agent]))
  # -----------------------------------------------

  # Testing agent
  # -----------------------------------------------
  done = False
  t = 0
  obs = env.reset()
  np.random.seed(seed)
  env.seed(seed)
  images = []

  if 'Billiard' in params.env_tag:
    saved_balls_pose = []
    saved_joints_pose = []
  elif 'Fastsim' in params.env_tag:
    saved_robot_pose = []
    obs = env.reset()
    pose = env.initPos
    p = fs.Posture(*pose)
    env.robot.set_pos(p)
  else:
    saved_robot_pose = []

  print()
  while not done:
    images.append(env.render(mode='rgb_array'))

    # agent_input = ts
    # action = utils.action_formatting(params.env_tag, selected['agent'](agent_input))

    if 'FastsimSimpleNavigation' in params.env_tag:
      agent_input = [obs, t / params.max_episode_len]  # Observation and time. The time is used to see when to stop the action. TODO move the action stopping outside of the agent
    elif 'Ant' in params.env_tag:
      agent_input = t
    else:
      agent_input = t/ params.max_episode_len

    action = utils.action_formatting(params.env_tag, selected['agent'](agent_input))

    obs, reward, done, info = env.step(action)
    t += 1
    if t >= params.max_episode_len:
      done = True

    if 'Ant' in params.env_tag:
      CoM = np.array([env.env.data.qpos[:2]])
      # CoM = np.array([self.env.robot.body_xyz[:2]])
      f_pose = CoM
      if np.any(np.abs(CoM) >= np.array([3, 3])):
        done = True
      saved_robot_pose.append(CoM)
    elif 'Billiard' in params.env_tag:
      saved_balls_pose.append(obs[0])
      saved_joints_pose.append(obs[1])
    elif 'Fastsim' in params.env_tag:
      saved_robot_pose.append(np.array(info['robot_pos'][:2]))

  images = np.array(images)
  with open('./maze_video_images3.npy', 'wb') as file:
    np.save(file, images)
  if 'Billiard' in params.env_tag:
    env.params.SHOW_ARM_IN_ARRAY = True
    saved_balls_pose = np.array(saved_balls_pose) * np.array([100., -100.]) + np.array([150., 150.])
    saved_joints_pose = np.array(saved_joints_pose)
    point_pose = np.array([np.sin(saved_joints_pose[:, 0]) + .9 * np.cos(np.pi / 2. - saved_joints_pose[:, 1] - saved_joints_pose[:, 0]),
                           np.cos(saved_joints_pose[:, 0]) + .9 * np.sin(np.pi / 2. - saved_joints_pose[:, 1] - saved_joints_pose[:, 0])]).transpose()
    point_pose = point_pose * np.array([-100., -100.]) + np.array([150., 300.])
  else:
    saved_robot_pose = np.array(saved_robot_pose)

  f_pose = utils.extact_hd_bs(env, obs, reward, done, info)

  state = env.render(mode='rgb_array')#, top_bottom=True)
  state = state / np.max((np.max(state), 1))

  plt.figure()
  if 'Billiard' in params.env_tag:
    for i in range(state.shape[0]):
      for j in range(state.shape[1]):
        if np.all(state[i, j] == np.zeros(3)):
          state[i, j] = np.ones(3)
    plt.imshow(state)
    plt.plot(saved_balls_pose[:, 0], saved_balls_pose[:, 1], 'r-')
    plt.plot(point_pose[:, 0], point_pose[:, 1], 'b-')
  elif 'Fastsim' in params.env_tag:
    state = 1 - state
    plt.imshow(state)
    saved_robot_pose = saved_robot_pose/3.
    plt.plot(saved_robot_pose[:, 0], saved_robot_pose[:, 1], 'r-')
  elif 'Ant' in params.env_tag:
    plt.imshow(state)
    saved_robot_pose = np.squeeze(saved_robot_pose)
    saved_robot_pose = saved_robot_pose * np.array([80, -80]) + np.array([250, 250])
    plt.plot(saved_robot_pose[:, 0], saved_robot_pose[:, 1], 'r-')


  final_distance = np.sqrt(np.sum((target_pose - f_pose) ** 2))
  print('Positional error: {}'.format(final_distance))
  plt.show()

  # # Testing
  # # -----------------------------------------------
  # print('Choose the target image:')
  # print('- A negative value to cycle between all the available targets')
  # print('- A number in range [0, 22]')
  # x_image = int(input(' '))
  # if x_image < 0:
  #   print('Testing all of the targets...')
  #   x_image = list(range(len(x)))
  # else:
  #   print("Testing on target {}".format(x_image))
  #   x_image = [x_image]
  #
  # for target in x_image:
  #   # Get new target BS point
  #   goal = torch.Tensor(x[target]).permute(2, 0, 1).unsqueeze(0).to(device)
  #   surprise, bs_point, reconstr = selector(goal/torch.max(torch.Tensor(np.array([torch.max(goal).cpu().data, 1]))))
  #   bs_point = bs_point.flatten().cpu().data.numpy()
  #   print('Target point surprise {}'.format(surprise.cpu().data))
  #   print('Target bs point {}'.format(bs_point))
  #
  #   # Get K closest agents
  #   # -----------------------------------------------
  #   bs_space = np.stack([a[0] for a in pop['features'].values])
  #
  #   # Get distances
  #   diff = np.atleast_2d(bs_space - bs_point)
  #   dists = np.sqrt(np.sum(diff * diff, axis=1))
  #   print('Min distance {}'.format(np.min(dists)))
  #
  #   k = 1 # Testing on just closest agent
  #   if len(dists) <= k:  # Should never happen
  #     idx = list(range(len(dists)))
  #     k = len(idx)
  #   else:
  #     idx = np.argpartition(dists, k)  # Get 15 nearest neighs
  #
  #   mean_k_dist = np.mean(dists[idx[:k]])
  #   print('Mean K distance {}'.format(mean_k_dist))
  #
  #   selected = pop[idx[:k]]
  #   print("Selected agent {}".format(idx[:k]))
  #   # -----------------------------------------------
  #
  #   # Testing agent
  #   # -----------------------------------------------
  #   done = False
  #   ts = 0
  #   obs = utils.obs_formatting(params.env_tag, env.reset())
  #   while not done:
  #     env.render()
  #     agent_input = ts
  #     action = utils.action_formatting(params.env_tag, selected.iloc[0]['agent'](agent_input))
  #     obs, reward, done, info = env.step(action)
  #     obs = utils.obs_formatting(params.env_tag, obs, reward, done, info)
  #     ts += 1
  #     CoM = np.array([env.env.data.qpos[:2]])
  #     if np.any(np.abs(CoM) >= np.array([3, 3])):
  #       done =True
  #
  #   state = env.render(mode='rgb_array')
  #   state = state/np.max((np.max(state), 1))
  #   fig, ax = plt.subplots(3)
  #   ax[0].imshow(state)
  #   ax[1].imshow(x[target])
  #   ax[2].imshow(reconstr.permute(0,2,3,1)[0].cpu().data)
  #   plt.show()
  #   # -----------------------------------------------
  # # -----------------------------------------------



