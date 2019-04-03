# Created by giuseppe
# Date: 03/04/19

import numpy as np
import matplotlib.pyplot as plt

def box_plot(data, title=None, subplot_title=None, xticks=None, ylabels=None, fontsize=12, rows=1, cols=1, box_colors=None, median_colors=None):
  total_plots = rows*cols
  assert len(data) == total_plots, 'Wrong dataset amount. Expected {} - Got {}'.format(total_plots, len(data))
  if subplot_title is not None:
    assert len(subplot_title) == total_plots, 'Wrong subplot titles amount. Expected {} - Got {}'.format(total_plots, len(subplot_title))
  if xticks is not None:
    assert len(xticks) == total_plots, 'Wrong xlabels amount. Expected {} - Got {}'.format(total_plots, len(xticks))
    for k in range(total_plots):
      assert len(data[k]) == len(xticks[k]), 'Data and xticks {} wrong dimensions. Data {} - Xticks {}'.format(k, len(data[k]), len(xticks[k]))
  if ylabels is not None:
    assert len(ylabels) == total_plots, 'Wrong xlabels amount. Expected {} - Got {}'.format(total_plots, len(ylabels))
  if box_colors is not None:
    assert len(data[0]) <= len(box_colors), 'Too much data for given colors. Data {} - Box Colors {}'.format(len(data[0]), len(box_colors))
  if median_colors is not None:
    assert len(data[0]) <= len(median_colors), 'Too much data for given colors. Data {} - Median Colors {}'.format(len(data[0]), len(median_colors))

  fig, axes = plt.subplots(nrows=rows, ncols=cols)
  boxplots = []

  for i in range(total_plots):
    boxplots.append(axes[i].boxplot(data[i], patch_artist=True))
    if subplot_title is not None:
      axes[i].set_title(subplot_title[i])
    if xticks is not None:
      axes[i].set_xticklabels(xticks[i], rotation=45, fontsize=fontsize)
    if ylabels is not None:
      axes[i].set_ylabel(ylabels[i], fontsize=fontsize)

  for bp in boxplots:
    for box, color in zip(bp['boxes'], box_colors):
      plt.setp(box, color=color, linewidth=2)

    for i, color in zip(list(range(0, len(bp['whiskers']), 2)), box_colors):
      plt.setp(bp['whiskers'][i], color=color, linewidth=2)
      plt.setp(bp['whiskers'][i + 1], color=color, linewidth=2)
      plt.setp(bp['caps'][i], color=color, linewidth=2)
      plt.setp(bp['caps'][i + 1], color=color, linewidth=2)

    for box, color in zip(bp['medians'], median_colors):
      plt.setp(box, color=color, linewidth=2)

  plt.figlegend(boxplots[0]['boxes'], xticks[0], fontsize=fontsize)
  if title is not None:
    fig.suptitle(title, fontsize=16)
  plt.show()


coverage = {'NS':[54.76, 54.2, 54.6, 55.92, 54.68],
            'AE Novelty 16': [51.88,52.32,50.64,53.32,53.64],
            'AE Novelty 32': [56.88,56.4,52.8,54.08,55.16],
            'AE Surprise': [23.48,21.04,21.24,24.76,21.8],
            'AE Pareto': [49.48,48.56,50.36,49.56,50.56],
            'AE Novelty Long': [74.84,72.12,72.84,66.92,68.64],
            'AE No Update': [51.16,45.08,40.8,49,41.08],
            'AE Pareto Arm': [42.04,41.96,37.44,38.96,30.04]}

solutions = {'NS':[2295,2284,2248,2309,2266],
            'AE Novelty 16': [2429,2437,2428,2436,2447],
            'AE Novelty 32': [2456,2452,2428,2438,2455],
            'AE Surprise': [2295,2318,2301,2328,2296],
            'AE Pareto': [2556,2473,2450,2655,2579],
            'AE Novelty Long': [4887,4881,4881,4916,4913],
            'AE No Update': [2449,2436,2428,2457,2433],
            'AE Pareto Arm': [3666,3997,3626,3629,3666]}

overlapping = {'NS':[40.34858388,40.67425569,39.27935943,39.45430922,39.67343336],
               'AE Novelty 16': [46.60354055,46.32745178,47.8583196,45.27914614,45.19820188],
               'AE Novelty 32': [42.1009772,42.4959217,45.63426689,44.54470878,43.82892057],
               'AE Surprise': [74.42265795,77.30802416,76.92307692,73.41065292,76.2630662],
               'AE Pareto': [51.60406886,50.90982612,48.6122449,53.33333333,50.98875533],
               'AE Novelty Long': [61.71475343,63.06084819,62.6920713,65.96826688,65.07225728],
               'AE No Update': [47.77460188,53.73563218,57.99011532,50.14245014,57.78873818],
               'AE Pareto Arm': [71.33115112,73.75531649,74.18643133,73.16065032,79.51445717]}

names = ['NS', 'AE Pareto', 'AE Pareto Arm']
coverage_metric_data = [coverage[key] for key in names]
solutions_metric_data = [solutions[key] for key in names]
overlapping_metric_data = [overlapping[key] for key in names]

box_colors = ['plum', 'royalblue', 'seagreen', 'lightcoral', 'khaki']
median_colors = ['purple', 'darkblue', 'darkgreen', 'firebrick', 'orange']

box_plot([coverage_metric_data, solutions_metric_data, overlapping_metric_data],
     title='Arm Presence',
     subplot_title=['Coverage', 'Solutions found', 'Overlapping'],
     xticks=[names, names, names],
     ylabels=['Percentage', 'Number', 'Percentage'],
     cols=3,
     box_colors=box_colors,
     median_colors=median_colors)

