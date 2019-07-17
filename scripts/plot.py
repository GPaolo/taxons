# Created by Giuseppe Paolo 
# Date: 02/07/2019

import numpy as np
import json
import os
import matplotlib.pyplot as plt

class GenPlot(object):

  def __init__(self, folders=None):
    self.folders = folders
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

  def load_exp_data(self, folder):
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
      coverage.append(np.array(list(map(np.float64, logs['Coverage']))))
      gen_surprise.append(np.array(list(map(np.float64, logs['Avg gen surprise']))))
      archive_size.append(np.array(list(map(int, logs['Archive size']))))
    coverage = np.array(coverage)
    gen_surprise = np.array(gen_surprise)
    archive_size = np.array(archive_size)
    gens = np.array(gens)
    return coverage, gen_surprise, archive_size, gens

  def plot_data(self, gens, data, title, labels, colors, y_axis):
    # Create plots
    fig, axes = plt.subplots(nrows=1, ncols=1)

    for exp, c, l in zip(data, colors, labels):
      std = np.std(exp, 0)
      mean = np.mean(exp, 0)
      max = np.max(exp, 0)
      min = np.min(exp, 0)

      axes.plot(gens, mean, color=c, label=l)
      axes.fill_between(gens, max, min, facecolor=c, alpha=0.3)

    axes.set_title(title)
    fig.legend(loc='upper left')
    axes.set_xlabel('Generations')
    axes.set_ylabel(y_axis)
    plt.show()

  def plot_data_single_fig(self, gens, data, title, labels, colors, y_axis, axes):
    for exp, c, l in zip(data, colors, labels):
      std = np.std(exp, 0)
      mean = np.mean(exp, 0)
      max = np.max(exp, 0)
      min = np.min(exp, 0)

      axes.plot(gens, mean, color=c, label=l)
      axes.fill_between(gens, max, min, facecolor=c, alpha=0.3)

    axes.set_title(title)
    axes.set_xlabel('Generations')
    axes.set_ylabel(y_axis)


if __name__ == '__main__':
  plotter = GenPlot()

  c_nt, s_nt, a_nt, g = plotter.load_exp_data('/home/giuseppe/src/rnd_qd/experiments/Billiard_AE_NoTrain')
  c_aen, s_aen, a_aen, g = plotter.load_exp_data('/home/giuseppe/src/rnd_qd/experiments/Billiard_AE_Novelty')
  c_aes, s_aes, a_aes, g = plotter.load_exp_data('/home/giuseppe/src/rnd_qd/experiments/Billiard_AE_Surprise')
  c_ps, s_ps, a_ps, g = plotter.load_exp_data('/home/giuseppe/src/rnd_qd/experiments/Billiard_PS')
  c_rs, s_rs, a_rs, g = plotter.load_exp_data('/home/giuseppe/src/rnd_qd/experiments/Billiard_RS')
  c_ns, s_ns, a_ns, g = plotter.load_exp_data('/home/giuseppe/src/rnd_qd/experiments/Billiard_NS')
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

  fig, axes = plt.subplots(nrows=1, ncols=3)
  plotter.plot_data_single_fig(g, [c_nt, c_ps, c_aen, c_aes, c_ns, c_rs],
                    labels=['NT', 'PS', 'AEN', 'AES', 'NS', 'RS'],
                    colors=['red', 'blue', 'yellow', 'green', 'violet', 'grey'],
                    title='Coverage', y_axis='Coverage %', axes=axes[0])
  plotter.plot_data_single_fig(g, [s_nt, s_ps, s_aen, s_aes, s_ns, s_rs],
                    labels=['NT', 'PS', 'AEN', 'AES', 'NS', 'RS'],
                    colors=['red', 'blue', 'yellow', 'green', 'violet', 'gray'],
                    title='Surprise', y_axis='Reconstruction error', axes=axes[1])
  plotter.plot_data_single_fig(g, [a_nt, a_ps, a_aen, a_aes, a_ns, a_rs],
                    labels=['NT', 'PS', 'AEN', 'AES', 'NS', 'RS'],
                    colors=['red', 'blue', 'yellow', 'green', 'violet', 'gray'],
                    title='Archive Size', y_axis='Number of agents', axes=axes[2])
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left')
  plt.subplots_adjust(left=0.1, right=.99, top=0.9, bottom=0.1, wspace=0.4)
  plt.show()