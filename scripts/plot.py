# Created by Giuseppe Paolo 
# Date: 02/07/2019

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from core.utils import utils

class GenPlot(object):

  def __init__(self, folders=None, total_gens=500):
    self.folders = folders
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    self.total_gens = total_gens

  def load_exp_data(self, folder):
    if not os.path.exists(folder):
      return None, None, None
    # Load data
    seeds = list(os.walk(folder))[0][1]
    coverage = []
    gen_surprise = []
    archive_size = []
    for seed in seeds:
      logs_path = os.path.join(folder, seed, 'logs.json')
      with open(logs_path) as f:
        logs = json.load(f)
      gens = list(map(int, logs['Generation']))
      if len(gens) < self.total_gens:
        print('Experiment {} Seed {} has {} gens'.format(folder, seed, len(gens)))
        continue
      coverage.append(np.array(list(map(np.float64, logs['Coverage']))))
      gen_surprise.append(np.array(list(map(np.float64, logs['Avg gen surprise']))))
      archive_size.append(np.array(list(map(int, logs['Archive size']))))
    coverage = np.array(coverage)
    gen_surprise = np.array(gen_surprise)
    archive_size = np.array(archive_size)
    gens = np.array(gens)
    if 'RS' in folder:
      return coverage, gen_surprise, archive_size
    if np.max(gens) < self.total_gens-1:
      return None, None, None
    return coverage, gen_surprise, archive_size

  def plot_data(self, data, title, labels, cmap, y_axis, gen=True):
    # Create plots
    if gen:
      x_data = np.array(list(range(self.total_gens)))
    else:
      x_data = np.array(list(range(0, self.total_gens*5, 5)))

    fig, axes = plt.subplots(nrows=1, ncols=1)
    colors = [cmap(k) for k in range(len(data))]
    axes.yaxis.grid(True)

    for exp, c, l in zip(data, colors, labels):
      std = np.std(exp, 0)
      mean = np.mean(exp, 0)
      max = np.max(exp, 0)
      min = np.min(exp, 0)

      # axes.grid(True)
      axes.plot(x_data, mean, color=c, label=l)
      axes.fill_between(x_data, max, min, facecolor=c, alpha=0.3)

    axes.set_title(title)
    fig.legend(loc='upper left')
    if gen:
      axes.set_xlabel('Search steps')
    else:
      axes.set_xlabel('Number of controllers')
    axes.set_ylabel(y_axis)
    plt.show()
    return fig

  def plot_data_single_fig(self, data, title, labels, cmap, y_axis, axes, use_std=False, gen=True):
    colors = [cmap(k) for k in range(len(data))]
    axes.yaxis.grid(True)
    if gen:
      x_data = np.array(list(range(self.total_gens)))
    else:
      x_data = np.array(list(range(0, self.total_gens*5, 5)))

    for d, c, l in zip(data, colors, labels):
      if d is None:
        continue
      experiment = d[:, :self.total_gens]
      std = np.std(experiment, 0)
      mean = np.mean(experiment, 0)
      if not use_std:
        max = np.max(experiment, 0)
        min = np.min(experiment, 0)
      else:
        max = mean + std
        min = mean - std

      # axes.grid(True)
      axes.plot(x_data, mean, color=c, label=l)
      axes.fill_between(x_data, max, min, facecolor=c, alpha=0.3)

    axes.set_title(title)
    if gen:
      axes.set_xlabel('Search steps')
    else:
      axes.set_xlabel('Number of controllers')
    axes.set_ylabel(y_axis)


if __name__ == '__main__':
  plotter = GenPlot(total_gens=2000)

  base_path = '/media/giuseppe/Storage/AE NS/Experiments/Billiard 2k'
  experiment = 'Billiard'

  c_mix, s_mix, a_mix = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Mixed'.format(experiment)))
  c_nt, s_nt, a_nt = plotter.load_exp_data(os.path.join(base_path,'{}_AE_NoTrain'.format(experiment)))
  c_nt, s_nt, a_nt = None, None, None
  c_aen, s_aen, a_aen = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Novelty'.format(experiment)))
  c_aes, s_aes, a_aes = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Surprise'.format(experiment)))
  c_ns, s_ns, a_ns = plotter.load_exp_data(os.path.join(base_path,'{}_NS'.format(experiment)))
  c_ps, s_ps, a_ps = plotter.load_exp_data(os.path.join(base_path,'{}_PS'.format(experiment)))
  c_rbd, s_rbd, a_rbd = plotter.load_exp_data(os.path.join(base_path,'{}_RBD'.format(experiment)))
  c_rs, s_rs, a_rs = plotter.load_exp_data(os.path.join(base_path,'{}_RS'.format(experiment)))

  # plotter.plot_data(g, [c_nt, c_ps, c_aen, c_aes, c_ns],
  #                   labels=['NT', 'PS', 'AEN', 'AES', 'NS'],
  #                   colors=['red', 'blue', 'yellow', 'green', 'violet'],
  #                   title='Coverage', y_axis='Coverage %')
  # plotter.plot_data(g, [s_nt, s_ps, s_aen, s_aes, s_ns],
  #                   labels=['NT', 'PS', 'AEN', 'AES', 'NS'],
  #                   colors=['red', 'blue', 'yellow', 'green', 'violet'],
  #                   title='Surprise', y_axis='Reconstruction error')
  # plotter.plot_data(g, [a_nt, a_ps, a_aen, a_aes, a_ns],
  #                   labels=['NT', 'PS', 'AEN', 'AES', 'NS'],
  #                   colors=['red', 'blue', 'yellow', 'green', 'violet'],
  #                   title='Archive Size', y_axis='Number of agents')

  use_std = True
  gen_on_x = False

  colors = plt.get_cmap('Set1')
  plt.rc('grid', linestyle="dotted", color='gray')
  labels = ['TAXONS', 'NT', 'AEN', 'AES', 'NS', 'PS', 'RBD', 'RS']
  coverage_list = [c_mix, c_nt, c_aen, c_aes, c_ns, c_ps, c_rbd, c_rs]
  surprise_list = [s_mix, s_nt, s_aen, s_aes, s_ns, s_ps, s_rbd, s_rs]
  archive_list = [a_mix, a_nt, a_aen, a_aes, a_ns, a_ps, a_rbd, a_rs]
  overlapping_list = []

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(60, 10))

  plotter.plot_data_single_fig(coverage_list,
                    labels=labels,
                    cmap=colors,
                    title='Coverage', y_axis='Coverage %', axes=axes[0],
                    use_std=use_std,
                    gen=gen_on_x)

  # plotter.plot_data_single_fig(surprise_list,
  #                  labels=labels,
  #                  cmap=colors,
  #                  title='Rec. Error', y_axis='Reconstruction error', axes=axes[1],
  #                  use_std = use_std,
  #                   gen=gen_on_x)

  for a, c in zip(archive_list, coverage_list):
    if a is not None:
      overlapping_list.append(utils.calc_avg_chi_sq_test((50, 50), a, c))
    else:
      overlapping_list.append(None)

  plotter.plot_data_single_fig(overlapping_list,
                    labels=labels,
                    cmap=colors,
                    title='Log Chi Squared', y_axis='Log distance from uniform', axes=axes[1],
                    use_std=use_std,
                    gen=gen_on_x)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left')
  plt.subplots_adjust(left=0.1, right=.99, top=0.9, bottom=0.1, wspace=0.4)
  plt.show()

  fig.savefig(os.path.join(base_path,'plots.pdf'))

  # plt.figure(figsize=(5, 10))