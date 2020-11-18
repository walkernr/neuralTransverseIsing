# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 05:51:23 2020

@author: Nicholas
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('font', family='sans-serif')
FTSZ = 28
FIGW = 16
PPARAMS = {'figure.figsize': (FIGW, FIGW),
            'lines.linewidth': 4.0,
            'legend.fontsize': FTSZ,
            'axes.labelsize': FTSZ,
            'axes.titlesize': FTSZ,
            'axes.linewidth': 2.0,
            'xtick.labelsize': FTSZ,
            'xtick.major.size': 20,
            'xtick.major.width': 2.0,
            'ytick.labelsize': FTSZ,
            'ytick.major.size': 20,
            'ytick.major.width': 2.0,
            'font.size': FTSZ}
plt.rcParams.update(PPARAMS)
SCALE = lambda a, b: (a-np.min(b))/(np.max(b)-np.min(b))
CM = plt.get_cmap('plasma')

def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-nm', '--name', help='simulation name',
                        type=str, default='init')
    parser.add_argument('-n', '--lattice_sites', help='lattice sites',
                        type=int, default=9)
    parser.add_argument('-t', '--time_slices', help='time slices',
                        type=int, default=27)
    args = parser.parse_args()
    return args.verbose, args.name, args.lattice_sites, args.time_slices

VERBOSE, NAME, N, P = parse_args()
# current working directory and prefix
CWD = os.getcwd()
PREF = CWD+'/%s.%d.%d' % (NAME, N, P)
# external fields and temperatures
H = np.load(PREF+'.h.npy')
T = np.load(PREF+'.t.npy')
NH, NT = H.size, T.size

DAT = np.load(PREF+'.dat.npy')

ENER = DAT[:, :, :, 0]
MAG = DAT[:, :, :, 1]

MENER = ENER.mean(2)
MMAG = MAG.mean(2)

SPHT = np.square(ENER.std(2))/np.square(T.reshape(1, -1))
MSUSC = np.square(MAG.std(2))/T.reshape(1, -1)


def plot_diagram(data, alias):
    file_name = PREF+'.{}.png'.format(alias)
    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('top', size='5%', pad=0.8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    im = ax.imshow(data, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    ax.grid(which='both', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(NT), minor=True)
    ax.set_yticks(np.arange(NH), minor=True)
    ax.set_xticks(np.arange(NT)[::4], minor=False)
    ax.set_yticks(np.arange(NH)[::4], minor=False)
    ax.set_xticklabels(np.round(T, 2)[::4], rotation=-60)
    ax.set_yticklabels(np.round(H, 2)[::4])
    # label axes
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\Gamma$')
    # place colorbal
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(data.min(), data.max(), 3))
    # save figure
    fig.savefig(file_name)
    plt.close()

plot_diagram(MENER, 'ener')
plot_diagram(MMAG, 'mag')
plot_diagram(SPHT, 'spht')
plot_diagram(MSUSC, 'msusc')