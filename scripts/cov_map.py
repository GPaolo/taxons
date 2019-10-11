 # Created by giuseppe
# Date: 03/04/19

import numpy as np
import gym
import gym_billiard, gym_fastsim, pybulletgym
import os
from scripts import parameters
from core.evolution import population, agents
from core.utils import utils
import pickle as pkl
import progressbar
import json
import matplotlib.pyplot as plt
import matplotlib
import gc
from sklearn.decomposition import PCA
import pandas as pd


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
          elif 'Ant' in self.params.env_tag:  # TODO metti questi anche nelle baselines
            agent_input = t
          else:
            agent_input = t / self.params.max_episode_len
          action = utils.action_formatting(self.params.env_tag, agent['agent'](agent_input))

          obs, reward, done, info = self.env.step(action)
          t += 1
          if t >= self.params.max_episode_len:
            done = True

          if 'Ant' in self.params.env_tag:
            CoM = np.array([self.env.env.data.qpos[:2]])#CoM = np.array([self.env.robot.body_xyz[:2]])
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
    if 'Fastsim' in self.params.env_tag:
      for k in range(len(bs)):
        bs[k][1] = 600. - bs[k][1]
    if 'Ant' in self.params.env_tag:
      for k in range(len(bs)):
        bs[k] = np.squeeze(bs[k])
    return bs
  # -----------------------------------------------

  # -----------------------------------------------
  def feat_pca(self):
    print('Doing PCA')
    feats = self.pop['features'].values
    feats = np.array([k[0] for k in feats])
    self.pca = PCA(n_components=2, whiten=False)
    pca_feat = self.pca.fit_transform(feats)
    print('Done.')
    return pca_feat
  # -----------------------------------------------

  # -----------------------------------------------
  def main(self, gen=1000, highlights=None, plot_coverage=True):
    print('Working on seed {}'.format(self.seed))
    folder_name = os.path.join(self.folder, self.seed)
    self.load_params(folder_name)
    self.load_logs(folder_name)
    self.load_archive(folder_name)
    self.env.seed(int(self.seed))
    np.random.seed(int(self.seed))
    self.env.reset()

    if True:# None in self.pop['bs'].values:
      self.evaluate_agent_xy()
      self.pop.save_pop(os.path.join(folder_name, 'models'), 'archive')

    if 'Ant' in self.params.env_tag:
      u_limit = 3.5
      l_limit = -u_limit
    elif 'FastsimSimpleNavigation' in self.params.env_tag:
      u_limit = 600
      l_limit = 0
    else:
      u_limit = 1.35
      l_limit = -u_limit
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

    bs = self.get_bs_by_gen(gen)
    if plot_coverage:
      fig, axes = plt.subplots(nrows=1, ncols=1)
      # Show actual policies
      self.show_bs_points(bs, axes=axes, lower_limit=l_limit, upper_limit=u_limit)
      if highlights is not None:
        self.show_bs_points(highlights, axes=axes, color='red', lower_limit=l_limit, upper_limit=u_limit)
      plt.show()
    else:
      pca_feat = self.feat_pca()
      pca_feat = pca_feat[:len(bs)]
      cmap = matplotlib.cm.get_cmap('viridis')
      normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(bs))
      colors = [cmap(normalize(value)) for value in range(len(bs))]
      bs_x = []
      bs_y = []
      feat_x = []
      feat_y = []
      for k in range(len(bs)):
        bs_x.append(bs[k][0])
        bs_y.append(bs[k][1])
        feat_x.append(pca_feat[k][0])
        feat_y.append(pca_feat[k][1])
      bs_x_M, bs_x_m = np.max(bs_x), np.min(bs_x)
      bs_y_M, bs_y_m = np.max(bs_y), np.min(bs_y)
      feat_x_M, feat_x_m = np.max(feat_x), np.min(feat_x)
      feat_y_M, feat_y_m = np.max(feat_y), np.min(feat_y)


      data = pd.DataFrame({'x': bs_x, 'y': bs_y, 'feat_x': feat_x, 'feat_y': feat_y})
      data.sort_values(['x', 'y'], ascending=True, inplace=True)
      data.reset_index(drop=True, inplace=True)
      fig, axes = plt.subplots(nrows=1, ncols=2)

      axes[0].set_xlim(bs_x_m, bs_x_M)
      axes[0].set_ylim(bs_y_m, bs_y_M)
      axes[1].set_xlim(feat_x_m, feat_x_M)
      axes[1].set_ylim(feat_y_m, feat_y_M)

      for k, c in zip(range(len(data)), colors):
        line = data.iloc[k]
        axes[0].scatter(line['x'], line['y'], c=c)
        axes[1].scatter(line['feat_x'], line['feat_y'], c=c)

      plt.show()


      # for x, y, fx, fy, c




    gc.collect()
  # -----------------------------------------------


if __name__ == "__main__":


  base_path = '/media/giuseppe/Storage/AE NS/Experiments/Ant/Ant_AE_Mixed/'
  seed = 15
  highlights = np.array([[2., 1.3],
                         [-1, -2],
                         [0.5, 2.2]
                         ])

  metric = CoverageMap(exp_folder=base_path, reeval_bs=True, render=False, seed=str(seed))
  metric.main(gen=500, highlights=highlights, plot_coverage=True)
