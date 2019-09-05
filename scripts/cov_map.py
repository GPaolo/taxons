 # Created by giuseppe
# Date: 03/04/19

import numpy as np
import gym
import gym_billiard, gym_fastsim, pybulletgym
import os
from scripts import parameters
from core.qd import population, agents
from core.utils import utils
import pickle as pkl
import progressbar
import json
import matplotlib.pyplot as plt
import matplotlib
import gc
from sklearn.decomposition import PCA


class CoverageMap(object):
  # -----------------------------------------------
  def __init__(self, exp_folder=None, reeval_bs=False, render=False, seed=11):
    assert os.path.exists(exp_folder), 'Experiment folder {} does not exist'.format(exp_folder)
    self.folder = exp_folder
    self.params = None
    self.reeval_bs = reeval_bs
    self.render_test = render

    # Get all the seeds
    # self.seeds = list(os.walk(self.folder))[0][1]
    self.seed = seed

    if 'Billiard' in self.folder:
      self.env_tag = 'Billiard-v0'
    elif 'Ant' in self.folder:
      self.env_tag = 'Ant-v2'
    elif 'Maze' in self.folder:
      self.env_tag = 'FastsimSimpleNavigation-v0'

    self.env = gym.make(self.env_tag)
    self.env.reset()
  # -----------------------------------------------

  # -----------------------------------------------
  def load_params(self, path):
    print('Loading parameters...')
    self.params = parameters.Params()
    self.params.load(os.path.join(path, 'params.json'))
    assert self.env_tag == self.params.env_tag, 'Env tag from folder different from parameters env tag: {} - {}'.format(
      self.env_tag, self.params.env_tag)
  # -----------------------------------------------

  # -----------------------------------------------
  def load_logs(self, path):
    logs_path = os.path.join(path, 'logs.json')
    with open(logs_path) as f:
      logs = json.load(f)
    gens = np.array(list(map(int, logs['Generation'])))
    coverage = np.array(list(map(np.float64, logs['Coverage'])))
    gen_surprise = np.array(list(map(np.float64, logs['Avg gen surprise'])))
    archive_size = np.array(list(map(int, logs['Archive size'])))
    self.exp_data = {'coverage': coverage,
                     'suprise': gen_surprise,
                     'archive_size': archive_size,
                     'generation': gens}
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
    print('Loaded "{} policies.'.format(len(self.pop)))
    self.pop.pop.sort_values('name', inplace=True)
    self.pop.pop.reset_index(drop=True, inplace=True)
  # -----------------------------------------------

  # -----------------------------------------------
  def evaluate_agent_xy(self):
    print('Calculating agent XY final pose')
    with progressbar.ProgressBar(max_value=len(self.pop)) as bar:
      for agent_idx, agent in enumerate(self.pop):
        done = False
        t = 0
        obs = self.env.reset()
        while not done:
          if self.render_test:
            self.env.render()
          if 'FastsimSimpleNavigation' in self.params.env_tag:
            agent_input = [obs, t / self.params.max_episode_len]  # Observation and time. The time is used to see when to stop the action. TODO move the action stopping outside of the agent
            action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input))
          else:
            agent_input = t
            action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input / self.params.max_episode_len))

          obs, reward, done, info = self.env.step(action)
          t += 1
          if t >= self.params.max_episode_len:
            done = True

          if 'Ant' in self.params.env_tag:
            CoM = np.array([self.env.robot.body_xyz[:2]])
            if np.any(np.abs(CoM) >= np.array([3, 3])):
              done = True

        agent['bs'] = utils.extact_hd_bs(self.env, obs, reward, done, info)
        bar.update(agent_idx)
  # -----------------------------------------------

  # -----------------------------------------------
  def show_bs_points(self, bs_points, upper_limit=1.35, lower_limit=-1.35, axes=None, color=None):
    pts = ([x[0] for x in bs_points if x is not None], [y[1] for y in bs_points if y is not None])

    plt.rcParams["patch.force_edgecolor"] = True
    if axes is None:
      fig, axes = plt.subplots(nrows=1, ncols=1)

    # axes.set_title('Final positions of agents'.format(len(pts[0])))
    if color is None:
      axes.scatter(pts[0], pts[1])
    else:
      axes.scatter(pts[0], pts[1], color=color)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_xlim(lower_limit, upper_limit)
    axes.set_ylim(lower_limit, upper_limit)
  # -----------------------------------------------

  # -----------------------------------------------
  def get_bs_by_gen(self, gen):
    if gen >= len(self.exp_data['archive_size']):
      gen = len(self.exp_data['archive_size']) - 1
    agents_num = self.exp_data['archive_size'][gen]
    bs = self.pop['bs'].values[:agents_num]
    return bs
  # -----------------------------------------------

  # -----------------------------------------------
  def feat_pca(self):
    print('Doing PCA')
    feats = self.pop['features'].values
    feats = np.array([k[0] for k in feats])
    self.pca = PCA(n_components=2, whiten=True)
    self.pca.fit(feats)
    print('Done.')
  # -----------------------------------------------

  # -----------------------------------------------
  def main(self, gen=1000, highlights=None):
    print('Working on seed {}'.format(self.seed))
    folder_name = os.path.join(self.folder, self.seed)
    self.load_params(folder_name)
    self.load_archive(folder_name)
    self.load_logs(folder_name)
    self.env.seed(int(self.seed))
    np.random.seed(int(self.seed))
    self.env.reset()

    if None in self.pop['bs'].values:
      self.evaluate_agent_xy()
      self.pop.save_pop(os.path.join(folder_name, 'models'), 'archive')
    #
    # pca_feat = self.feat_pca()
    #
    # cmap = matplotlib.cm.get_cmap('viridis')
    # bs = self.get_bs_by_gen(gen)
    # normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(bs))
    # colors = [cmap(normalize(value)) for value in range(len(bs))]
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # self.show_bs_points(bs, axes=axes[0], color=colors)
    # self.show_bs_points(pca_feat, axes=axes[1], color=colors)
    # plt.show()

      #gens = 0
    fig, axes = plt.subplots(nrows=1, ncols=1)
      #plt.ion()
      #plt.show()
      #while gens >= 0:
      #  gens = int(input('Number of gens to show '))
    bs = self.get_bs_by_gen(gen)
    # Show actual policies
    self.show_bs_points(bs, axes=axes)
    if highlights is not None:
      self.show_bs_points(highlights, axes=axes, color='red')
      # bs = self.get_bs_by_gen(300)
      # self.show_bs_points(bs, axes=axes[1])
      #  plt.draw()
      #  plt.pause(.01)
      #gens = [[10, 50, 150], [300, 600, 999]]
      #for r in range(2):
      #  for c in range(3):
      #    bs = self.get_bs_by_gen(gens[r][c])
      #    self.show_bs_points(bs, axes=axes[r][c])
    plt.show()

    gc.collect()
  # -----------------------------------------------


if __name__ == "__main__":


  base_path = '/home/giuseppe/src/rnd_qd/experiments/Billiard_AE_Mixed'
  seed = 11
  highlights = np.array([[1, -0.8],
                         [0, 0.5],
                         [1.1, 1.1]
                         ])

  metric = CoverageMap(exp_folder=base_path, reeval_bs=True, render=False, seed=str(seed))
  metric.main(gen=2000, highlights=highlights)
