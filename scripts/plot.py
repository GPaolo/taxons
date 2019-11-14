# Created by Giuseppe Paolo 
# Date: 02/07/2019

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from core.utils import utils
from scipy.stats import mannwhitneyu
from pprint import pprint

class GenPlot(object):

  def __init__(self, folders=None, total_gens=500, cmaps=['Dark2']):
    self.folders = folders
    params = {'legend.fontsize': 'x-large',
#              'figure.figsize': (15, 5),
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    # plt.rcParams.update(params)

    self.total_gens = total_gens
    self.cmaps = [plt.get_cmap(map) for map in cmaps]


  def load_exp_data(self, folder):
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

    for i in range(len(seeds)):
      coverage[i] = coverage[i][:max_gens]
      gen_surprise[i] = gen_surprise[i][:max_gens]
      archive_size[i] = archive_size[i][:max_gens]

    coverage = np.array(coverage)
    gen_surprise = np.array(gen_surprise)
    archive_size = np.array(archive_size)
    gens = np.array(list(range(max_gens)))
    if 'RS' in folder:
      return coverage, gen_surprise, archive_size
    if max_gens < self.total_gens-1:
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

  def get_colors(self, data_size):
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
    #colors.append(cmap(0.2))
    #colors.append(cmap(0.0))
    #colors.append(cmap(0.45))
    #colors.append(cmap(0.9))
    colors.append(cmap(3))
    colors.append(cmap(1))
    colors.append(cmap(2))
    colors.append(cmap(0))
    colors.append(cmap(7))


    return colors

  def plot_curves(self, data, title, labels, y_axis, axes, use_std=False, gen=True):
    colors = self.get_colors(len(data))
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

      if l == 'TAXONS':
        linestyle = '-'
        linewidth = 2
      elif len(l) >= 4:
        linestyle = '-'
        linewidth = 2
      else:
        linestyle = '-.'
        linewidth = 2

      axes.plot(x_data, mean, color=c, label=l, linestyle=linestyle, linewidth=linewidth)
      axes.fill_between(x_data, max, min, facecolor=c, alpha=0.15)

    axes.set_title(title)
    if gen:
      axes.set_xlabel('Search steps')
    else:
      axes.set_xlabel('Number of controllers')
    axes.set_ylabel(y_axis)

  def plot_violins(self, data, title, labels, y_axis, axes):
    colors = self.get_colors(len(data))
    axes.grid(True)
    axes.grid(linestyle='dotted')

    violin_data = []
    pos = []
    violin_colors = []
    # Reformat data
    for d, l, c in zip(data, labels, colors):
      if d is None:
        continue
      print(l)
      print(np.mean(np.array(d[:,-1])))

      violin_data.append(d[:,-1])
      pos.append(l)
      violin_colors.append(list(c))

    print(np.mean(np.array(violin_data)))

    parts = axes.violinplot(violin_data,
                            showmeans=False,
                            showmedians=False,
                            showextrema=False)

    axes.get_xaxis().set_tick_params(direction='out',labelrotation=45)
    axes.xaxis.set_ticks_position('bottom')
    axes.set_xticks(np.arange(1, len(pos) + 1))
    axes.set_xticklabels(pos)
    axes.set_xlim(0.25, len(pos) + 0.75)

    for body, c in zip(parts['bodies'], violin_colors):
      c[-1] = 0.3
      body.set_facecolor(c)
      c[-1] = 1
      body.set_edgecolor(c)

    # def adjacent_values(vals, q1, q3):
    #   upper_adjacent_value = q3 + (q3 - q1) * 1.5
    #   upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    #
    #   lower_adjacent_value = q1 - (q3 - q1) * 1.5
    #   lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    #   return lower_adjacent_value, upper_adjacent_value

    quartile1 = []
    quartile3 = []
    medians = []
    for vd in violin_data:
      q1, m, q3 = np.percentile(vd, [25, 50, 75], axis=0)
      quartile1.append(q1)
      quartile3.append(q3)
      medians.append(m)


    whiskers = []
    for d in violin_data:
      whiskers.append([min(d), max(d)])
    # whiskers = np.array([adjacent_values(sorted_array, q1, q3)
    #   for sorted_array, q1, q3 in zip(violin_data, quartile1, quartile3)])
    # whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]



    inds = np.arange(1, len(medians) + 1)
    for idx, m, q1, q3, c, w in  zip(inds, medians, quartile1, quartile3, violin_colors, whiskers):
      # axis.vlines(idx, w[0], w[1], color='k', linestyle='-', lw=1)
      # axis.vlines(idx, w[0] - 0.15, w[0] + 0.15, color='k', linestyle='-', lw=6)
      # axis.vlines(idx, w[1] - 0.15, w[1] + 0.15, color='k', linestyle='-', lw=10)
      axes.vlines(idx, m-0.15, m+0.12, color=c, alpha=1, linestyle='-', lw=20)
      axes.vlines(idx, q1, q3, color=c, linestyle='-', lw=5, alpha=1)

    axes.set_ylabel(y_axis)
    axes.set_title(title)

  def mwu(self, coverage_data, names):
    results = {}
    for i in range(len(coverage_data)):
      if coverage_data[i] is None:
        continue
      for j in range(i+1, len(coverage_data)):
        if coverage_data[j] is None:
          continue
        x = coverage_data[i][:, -1]
        y = coverage_data[j][:, -1]
        name_x = names[i]
        name_y = names[j]
        name = '{}_{}'.format(name_x, name_y)
        results[name] = mannwhitneyu(x, y)
    return results

  def holm_bonferroni(self, p_values):
    from multipy.fwer import sidak
    names = []
    pvals = []
    for name in p_values:
      names.append(name)
      pvals.append(p_values[name][1])

    significant_pvals = sidak(pvals, alpha=0.05)
    print([k for k in zip(['{}: {:.4f}'.format(name, p) for name, p in zip(names, pvals)], significant_pvals)])


if __name__ == '__main__':
  total_gens = [1999, 1000, 500]
  violins = True
  fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,3.5))
  name = ['Billiard', 'Maze', 'Ant']
  name = ['Billiard', 'Maze']

  for experiment, ax, gens in zip(name, axes, total_gens):
    base_path = '/mnt/7e0bad1b-406b-4582-b7a1-84327ae60fc4/ICRA 2020/Experiments/{}'.format(experiment)

    plotter = GenPlot(total_gens=gens, cmaps=['Dark2','gist_rainbow'])

    c_mix, s_mix, a_mix = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Mixed'.format(experiment)))
    c_nt, s_nt, a_nt = plotter.load_exp_data(os.path.join(base_path,'{}_AE_NoTrain'.format(experiment)))
    # c_nt, s_nt, a_nt = None, None, None
    c_aen, s_aen, a_aen = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Novelty'.format(experiment)))
    c_aes, s_aes, a_aes = plotter.load_exp_data(os.path.join(base_path,'{}_AE_Surprise'.format(experiment)))
    c_ns, s_ns, a_ns = plotter.load_exp_data(os.path.join(base_path,'{}_NS'.format(experiment)))
    c_ps, s_ps, a_ps = plotter.load_exp_data(os.path.join(base_path,'{}_PS'.format(experiment)))
    c_rbd, s_rbd, a_rbd = plotter.load_exp_data(os.path.join(base_path,'{}_RBD'.format(experiment)))
    c_rs, s_rs, a_rs = plotter.load_exp_data(os.path.join(base_path,'{}_RS'.format(experiment)))
    c_ibd, s_ibd, a_ibd = plotter.load_exp_data(os.path.join(base_path,'{}_IBD'.format(experiment)))


    use_std = True
    gen_on_x = False

    plt.rc('grid', linestyle="dotted", color='gray')
    labels = ['PS', 'RBD', 'RS', 'NS', 'NT', 'IBD', 'TAXO-N', 'TAXO-S', 'TAXONS']
    coverage_list = [c_ps, c_rbd, c_rs, c_ns, c_nt, c_ibd, c_aen, c_aes, c_mix]
    surprise_list = [s_ps, s_rbd, s_rs, s_ns, s_nt, s_ibd, s_aen, s_aes, s_mix]
    archive_list = [a_ps, a_rbd, a_rs, a_ns, a_nt, a_ibd, a_aen, a_aes, a_mix]
    overlapping_list = []

    mwu = plotter.mwu(coverage_list, labels)
    plotter.holm_bonferroni(mwu)
    print('MWU for: {}\n'.format(experiment))
    # for k in mwu:
    #   print('{}:\n statistics: {} \n pvalue: {}\n'.format(k, mwu[k][0], mwu[k][1]))
    # print()


    if not violins:
      plotter.plot_curves(coverage_list,
                          labels=labels,
                          title=experiment, y_axis='Coverage %', axes=ax,
                          use_std=use_std,
                          gen=gen_on_x)
    else:
      plotter.plot_violins(coverage_list,
                          labels=labels,
                          title=experiment, y_axis='Coverage %', axes=ax)


  # plotter.plot_violins(coverage_list,
  #                      labels=labels,
  #                      cmap=colors,
  #                      title='Coverage',
  #                      y_axis='Coverage %',
  #                      axis=axes[1])

  # plotter.plot_data_single_fig(surprise_list,
  #                  labels=labels,
  #                  cmap=colors,
  #                  title='Rec. Error', y_axis='Reconstruction error', axes=axes[1],
  #                  use_std = use_std,
  #                   gen=gen_on_x)

  # for a, c in zip(archive_list, coverage_list):
  #   if a is not None:
  #     overlapping_list.append(utils.calc_avg_chi_sq_test((50, 50), a, c))
  #   else:
  #     overlapping_list.append(None)
  #
  # plotter.plot_data_single_fig(overlapping_list,
  #                   labels=labels,
  #                   cmap=colors,
  #                   title='Log Chi Squared', y_axis='Log distance from uniform', axes=axes[1],
  #                   use_std=use_std,
  #                   gen=gen_on_x)

  if not violins:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', fancybox=True)#, bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.12, right=.99, top=0.91, bottom=0.18, wspace=0.13)

  else:
    plt.subplots_adjust(left=0.05, right=.99, top=0.91, bottom=0.18, wspace=0.13)
  plt.show()

  # fig.savefig(os.path.join(base_path,'plots.pdf'))

  # plt.figure(figsize=(5, 10))