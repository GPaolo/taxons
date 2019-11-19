# Created by Giuseppe Paolo 
# Date: 02/07/2019

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

class GenPlot(object):

  # ---------------------------------------------------
  def __init__(self, total_gens=500):
    """
    Constructor
    :param total_gens: Total generations to plot
    """
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}

    self.total_gens = total_gens
    self._get_colors()
  # ---------------------------------------------------

  # ---------------------------------------------------
  def load_exp_data(self, folder):
    """
    Function to load experimental data to plot
    :param folder: Folder path in which the data are
    :return: If path does not exist: None, None, None. If the path exists: coverage, gen_surprise, archive_size
    """

    if not os.path.exists(folder):
      return None, None, None

    # Load data
    seeds = list(os.walk(folder))[0][1]
    coverage = []
    gen_surprise = []
    archive_size = []
    max_gens = self.total_gens

    for seed in seeds:
      logs_path = os.path.join(folder, seed, 'logs.json')
      with open(logs_path) as f:
        logs = json.load(f)
      gens = list(map(int, logs['Generation']))
      if len(gens) < max_gens:
        print('Experiment {} Seed {} has {} gens'.format(folder, seed, len(gens)))
        max_gens = len(gens)
        continue
      coverage.append(np.array(list(map(np.float64, logs['Coverage']))))
      gen_surprise.append(np.array(list(map(np.float64, logs['Avg gen surprise']))))
      archive_size.append(np.array(list(map(int, logs['Archive size']))))

    # Trim list of datas to the max_gens
    for i in range(len(seeds)):
      coverage[i] = coverage[i][:max_gens]
      gen_surprise[i] = gen_surprise[i][:max_gens]
      archive_size[i] = archive_size[i][:max_gens]

    coverage = np.array(coverage)
    gen_surprise = np.array(gen_surprise)
    archive_size = np.array(archive_size)

    if max_gens < self.total_gens-1:
      return None, None, None
    return coverage, gen_surprise, archive_size
  # ---------------------------------------------------

  # ---------------------------------------------------
  def _get_colors(self):
    """
    This function extract the colors from the colormaps
    :return:
    """
    colors = []
    # For baselines
    cmap = plt.get_cmap('Dark2')
    colors.append(cmap(5))
    colors.append(cmap(6))
    colors.append(cmap(7))
    colors.append(cmap(2))
    colors.append(cmap(1))

    # For TAXO
    cmap = plt.get_cmap('Set1')
    colors.append(cmap(3))
    colors.append(cmap(1))
    colors.append(cmap(2))
    colors.append(cmap(0))
    colors.append(cmap(7))

    self.colors = colors
  # ---------------------------------------------------

  # ---------------------------------------------------
  def plot_curves(self, data, title, labels, y_axis, axes, use_std=False, gen=True):
    """
    Plot the curves of the data
    :param data: List of data lists. Dimensions are: [method, seed, gen]
    :param title: Title of the graph
    :param labels: Labels of each line
    :param y_axis: Label of y axis
    :param axes: Pyplot axes
    :param use_std: Flag to choose if to use std deviation or min_max for the error bands
    :param gen: Generations or number of agents along the x axis
    """
    axes.yaxis.grid(True)
    if gen:
      x_data = np.array(list(range(self.total_gens)))
    else:
      x_data = np.array(list(range(0, self.total_gens*5, 5)))

    for d, c, l in zip(data, self.colors, labels):
      if d is None: # If no data for the exp, skip it
        continue
      experiment = d[:, :self.total_gens] # Select max gens to plot

      std = np.std(experiment, 0)
      mean = np.mean(experiment, 0)
      if not use_std:
        max = np.max(experiment, 0)
        min = np.min(experiment, 0)
      else:
        max = mean + std
        min = mean - std

      # Select line styles
      if l == 'TAXONS':
        linestyle = '-'
        linewidth = 2
      elif len(l) >= 4:
        linestyle = '-'
        linewidth = 2
      else:
        linestyle = '-.'
        linewidth = 2

      axes.plot(x_data, mean, color=c, label=l, linestyle=linestyle, linewidth=linewidth) # Plot mean
      axes.fill_between(x_data, max, min, facecolor=c, alpha=0.15) # Plot error bars

    axes.set_title(title)
    if gen:
      axes.set_xlabel('Search steps')
    else:
      axes.set_xlabel('Number of controllers')
    axes.set_ylabel(y_axis)
  # ---------------------------------------------------

  # ---------------------------------------------------
  def plot_violins(self, data, title, labels, y_axis, axes):
    """
    Plot violins plots
    :param data: List of data lists. Dimensions are: [method, seed, gen]
    :param title: Graph title
    :param labels: Labels of each line
    :param y_axis: Label of y axis
    :param axes: Pyplot axes
    """
    axes.grid(True)
    axes.grid(linestyle='dotted')

    violin_data = []
    pos = []
    violin_colors = []

    # Reformat data
    for d, l, c in zip(data, labels, self.colors):
      if d is None:  # If no data for the exp, skip it
        continue

      violin_data.append(d[:,-1])
      pos.append(l)
      violin_colors.append(list(c))

    parts = axes.violinplot(violin_data,
                            showmeans=False,
                            showmedians=False,
                            showextrema=False)

    # Format x axis
    axes.get_xaxis().set_tick_params(direction='out',labelrotation=45)
    axes.xaxis.set_ticks_position('bottom')
    axes.set_xticks(np.arange(1, len(pos) + 1))
    axes.set_xticklabels(pos)
    axes.set_xlim(0.25, len(pos) + 0.75)

    # Set colors for violins
    for body, c in zip(parts['bodies'], violin_colors):
      c[-1] = 0.3
      body.set_facecolor(c)
      c[-1] = 1
      body.set_edgecolor(c)

    # Calculate quartiles and medians
    quartile1 = []
    quartile3 = []
    medians = []
    for vd in violin_data:
      q1, m, q3 = np.percentile(vd, [25, 50, 75], axis=0)
      quartile1.append(q1)
      quartile3.append(q3)
      medians.append(m)

    # Calculate whiskers
    whiskers = []
    for d in violin_data:
      whiskers.append([min(d), max(d)])

    # Plot violins
    inds = np.arange(1, len(medians) + 1)
    for idx, m, q1, q3, c, w in  zip(inds, medians, quartile1, quartile3, violin_colors, whiskers):
      axes.vlines(idx, m-0.15, m+0.12, color=c, alpha=1, linestyle='-', lw=20)
      axes.vlines(idx, q1, q3, color=c, linestyle='-', lw=5, alpha=1)

    axes.set_ylabel(y_axis)
    axes.set_title(title)
  # ---------------------------------------------------

  # ---------------------------------------------------
  def mwu(self, coverage_data, names):
    """
    Mann-Whitney U test calculator
    :param coverage_data: Data on which to calculate the test
    :param names: Methods names
    :return:
    """
    results = {}

    for i in range(len(coverage_data)):
      if coverage_data[i] is None: # Skip if no coverage data is present
        continue
      for j in range(i+1, len(coverage_data)):
        if coverage_data[j] is None: # Skip if no coverage data is present
          continue
        x = coverage_data[i][:, -1]
        y = coverage_data[j][:, -1]
        name_x = names[i]
        name_y = names[j]
        name = '{}_{}'.format(name_x, name_y)
        results[name] = mannwhitneyu(x, y)

    return results
  # ---------------------------------------------------

  # ---------------------------------------------------
  def holm_bonferroni(self, p_values):
    """
    Holm-Bonferroni method calculator. Prints the correlation between the methods
    :param p_values: MWU p values
    """
    from multipy.fwer import sidak
    names = []
    pvals = []
    for name in p_values:
      names.append(name)
      pvals.append(p_values[name][1])

    significant_pvals = sidak(pvals, alpha=0.05)
    print([k for k in zip(['{}: {:.4f}'.format(name, p) for name, p in zip(names, pvals)], significant_pvals)])
  # ---------------------------------------------------


if __name__ == '__main__':

  save_plots = False
  curves_fig, curves_axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,3.5))
  violins_fig, violins_axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,3.5))

  name = ['Billiard', 'Maze', 'Ant'] # Experiments names
  total_gens = [1999, 1000, 500] # Generations per experiments to be plotted

  for experiment, c_ax, v_ax, gens in zip(name, curves_axes, violins_axes, total_gens):
    base_path = '/mnt/7e0bad1b-406b-4582-b7a1-84327ae60fc4/ICRA 2020/Experiments/{}'.format(experiment) # Data path
    use_std = True
    gen_on_x = False

    plotter = GenPlot(total_gens=gens)
    # Load data
    c_mix, s_mix, a_mix = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Mixed'.format(experiment)))
    c_nt, s_nt, a_nt = plotter.load_exp_data(os.path.join(base_path,'{}_AE_NoTrain'.format(experiment)))
    c_aen, s_aen, a_aen = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Novelty'.format(experiment)))
    c_aes, s_aes, a_aes = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Surprise'.format(experiment)))
    c_ns, s_ns, a_ns = plotter.load_exp_data(os.path.join(base_path,'{}_NS'.format(experiment)))
    c_ps, s_ps, a_ps = plotter.load_exp_data(os.path.join(base_path,'{}_PS'.format(experiment)))
    c_rbd, s_rbd, a_rbd = plotter.load_exp_data(os.path.join(base_path,'{}_RBD'.format(experiment)))
    c_rs, s_rs, a_rs = plotter.load_exp_data(os.path.join(base_path,'{}_RS'.format(experiment)))
    c_ibd, s_ibd, a_ibd = plotter.load_exp_data(os.path.join(base_path,'{}_IBD'.format(experiment)))

    # Arrange loaded data
    plt.rc('grid', linestyle="dotted", color='gray')
    labels = ['PS', 'RBD', 'RS', 'NS', 'NT', 'IBD', 'TAXO-N', 'TAXO-S', 'TAXONS']
    coverage_list = [c_ps, c_rbd, c_rs, c_ns, c_nt, c_ibd, c_aen, c_aes, c_mix]
    surprise_list = [s_ps, s_rbd, s_rs, s_ns, s_nt, s_ibd, s_aen, s_aes, s_mix]
    archive_list = [a_ps, a_rbd, a_rs, a_ns, a_nt, a_ibd, a_aen, a_aes, a_mix]
    overlapping_list = []

    # MWU and HB tests
    mwu = plotter.mwu(coverage_list, labels)
    plotter.holm_bonferroni(mwu)
    print('MWU for: {}\n'.format(experiment))

    plotter.plot_curves(coverage_list,
                        labels=labels,
                        title=experiment, y_axis='Coverage %', axes=c_ax,
                        use_std=use_std,
                        gen=gen_on_x)

    plotter.plot_violins(coverage_list,
                         labels=labels,
                         title=experiment, y_axis='Coverage %', axes=v_ax)

  # Curves adjust and legend plot
  handles, labels = curves_axes[0].get_legend_handles_labels()
  curves_fig.legend(handles, labels, loc='upper left', fancybox=True)#, bbox_to_anchor=(1, 0.5))
  curves_fig.subplots_adjust(left=0.12, right=.99, top=0.91, bottom=0.18, wspace=0.13)

  # Violins adjust
  violins_fig.subplots_adjust(left=0.05, right=.99, top=0.91, bottom=0.18, wspace=0.13)
  plt.show()

  if save_plots:
    violins_fig.savefig(os.path.join(base_path,'violins.pdf'))
    curves_fig.savefig(os.path.join(base_path,'curves.pdf'))
