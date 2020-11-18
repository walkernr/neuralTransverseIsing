# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:45:15 2020

@author: Nicholas
"""

import argparse
import os
import numpy as np

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

DAT = np.loadtxt(PREF+'.dat', dtype=np.float32).reshape(NH, NT, -1, 3)
DMP = np.loadtxt(PREF+'.dmp', dtype=np.int8).reshape(NH, NT, -1, N, P)

if VERBOSE:
    print('all data loaded')

np.save(PREF+'.dat.npy', DAT)
np.save(PREF+'.dmp.npy', DMP)

if VERBOSE:
    print('all data dumped')
