import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


plt.rcParams.update({'font.size': 20})

def plot_mape(gt, pred, title, fname, maxy=500, maxx=0.30):
    ape = np.abs(gt - pred) / np.abs(gt)
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 10))
    sns.distplot(ape, bins=np.linspace(0, 1, 400), norm_hist=False, kde=False)
    plt.axvline(ape.mean(), 0, len(gt), linewidth=3)
    
    q95 = np.percentile(ape, 95)
    plt.axvline(q95, 0, len(gt), linewidth=3, color='red')
    
    plt.legend(['Mean', '95%', 'Distribution'])
    plt.xticks(np.linspace(0, 1, 41), labels=['{}%'.format(p) for p in np.linspace(0, 100, 41)]) 
    plt.xlim(0, maxx)
    plt.ylim(0, maxy)
    plt.xlabel('Error rate')
    plt.ylabel('Samples number')
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_sizes(sizes, title, fname, maxy=500, maxx=0.30):
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 10))
    sns.distplot(sizes, bins=np.linspace(0, 100, 100), norm_hist=False, kde=False)
    plt.axvline(np.mean(sizes), 0, len(sizes), linewidth=3)
    
#     q95 = np.percentile(ape, 95)
#     plt.axvline(q95, 0, len(gt), linewidth=3, color='red')
    
    plt.legend(['Mean', 'Distribution'])
#     plt.xticks(np.linspace(0, 1, 41), labels=['{}%'.format(p) for p in np.linspace(0, 100, 41)]) 
    plt.xlim(0, maxx)
    plt.ylim(0, maxy)
    plt.xlabel('Side length')
    plt.ylabel('Samples number')
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    

def plot_targets(targets, title, fname, maxy=500, maxx=0.30):
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 10))
    sns.distplot(targets, bins=np.linspace(0, maxx, 100), norm_hist=False, kde=False)
    plt.axvline(np.mean(targets), 0, len(targets), linewidth=3)
    
#     q95 = np.percentile(ape, 95)
#     plt.axvline(q95, 0, len(gt), linewidth=3, color='red')
    
    plt.legend(['Mean', 'Distribution'])
#     plt.xticks(np.linspace(0, 1, 41), labels=['{}%'.format(p) for p in np.linspace(0, 100, 41)]) 
    plt.xlim(0, maxx)
    plt.ylim(0, maxy)
    plt.xlabel('Hydraulic conductance')
    plt.ylabel('Samples number')
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()