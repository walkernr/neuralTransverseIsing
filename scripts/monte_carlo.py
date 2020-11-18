# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:54:06 2020

@author: Nicholas
"""

import argparse
import os
import time
import numpy as np
import numba as nb
import itertools as it
from tqdm import tqdm

# --------------
# run parameters
# --------------


def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output mode', action='store_true')
    parser.add_argument('-r', '--restart', help='restart run mode', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run mode', action='store_true')
    parser.add_argument('-c', '--client', help='dask client run mode', action='store_true')
    parser.add_argument('-d', '--distributed', help='distributed run mode', action='store_true')
    parser.add_argument('-rd', '--restart_dump', help='restart dump frequency',
                        type=int, default=128)
    parser.add_argument('-rn', '--restart_name', help='restart dump simulation name',
                        type=str, default='ising_init')
    parser.add_argument('-rs', '--restart_step', help='restart dump start step',
                        type=int, default=512)
    parser.add_argument('-q', '--queue', help='job submission queue',
                        type=str, default='jobqueue')
    parser.add_argument('-a', '--allocation', help='job submission allocation',
                        type=str, default='startup')
    parser.add_argument('-nn', '--nodes', help='job node count',
                        type=int, default=1)
    parser.add_argument('-np', '--procs_per_node', help='number of processors per node',
                        type=int, default=12)
    parser.add_argument('-w', '--walltime', help='job walltime',
                        type=int, default=72)
    parser.add_argument('-m', '--memory', help='job memory (total)',
                        type=int, default=32)
    parser.add_argument('-nw', '--workers', help='job worker count (total)',
                        type=int, default=12)
    parser.add_argument('-nt', '--threads', help='threads per worker',
                        type=int, default=1)
    parser.add_argument('-mt', '--method', help='parallelization method',
                        type=str, default='fork')
    parser.add_argument('-nm', '--name', help='simulation name',
                        type=str, default='init')
    parser.add_argument('-n', '--lattice_sites', help='lattice sites',
                        type=int, default=27)
    parser.add_argument('-t', '--time_slices', help='time slices',
                        type=int, default=81)
    parser.add_argument('-j', '--interaction', help='interaction j',
                        type=float, default=1.0)
    parser.add_argument('-hn', '--field_number', help='number of external fields',
                        type=int, default=16)
    parser.add_argument('-hr', '--field_range', help='field range (low and high)',
                        type=float, nargs=2, default=[0.1, 8.1])
    parser.add_argument('-tn', '--temperature_number', help='number of temperatures',
                        type=int, default=16)
    parser.add_argument('-tr', '--temperature_range', help='temperature range (low and high)',
                        type=float, nargs=2, default=[0.1, 8.1])
    parser.add_argument('-sc', '--sample_cutoff', help='sample recording cutoff',
                        type=int, default=128)
    parser.add_argument('-sn', '--sample_number', help='number of samples to generate',
                        type=int, default=640)
    parser.add_argument('-rec', '--remcmc_cutoff', help='replica exchange markov chain monte carlo cutoff',
                        type=int, default=128)
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return (args.verbose, args.parallel, args.client, args.distributed, args.restart,
            args.restart_dump, args.restart_name, args.restart_step,
            args.queue, args.allocation, args.nodes, args.procs_per_node,
            args.walltime, args.memory,
            args.workers, args.threads, args.method,
            args.name, args.lattice_sites, args.time_slices,
            args.interaction,
            args.field_number, *args.field_range,
            args.temperature_number, *args.temperature_range,
            args.sample_cutoff, args.sample_number, args.remcmc_cutoff)


def client_info():
    ''' print client info '''
    info = str(CLIENT.scheduler_info)
    info = info.replace('<', '').replace('>', '').split()[6:8]
    print('\n%s %s' % tuple(info))

# -----------------------------
# output file utility functions
# -----------------------------


def file_prefix():
    ''' returns filename prefix for simulation '''
    prefix = os.getcwd()+'/%s.%d.%d' % (NAME, N, P)
    return prefix


def init_output(k):
    ''' initializes output filenames for a sample '''
    # extract field/temperature indices from index
    i, j = np.unravel_index(k, shape=(NH, NT), order='C')
    dat = file_prefix()+'.%02d.%02d.dat' % (i, j)
    dmp = file_prefix()+'.%02d.%02d.dmp' % (i, j)
    # clean old output files if they exist
    if os.path.isfile(dat):
        os.remove(dat)
    if os.path.isfile(dmp):
        os.remove(dmp)
    return dat, dmp


def init_outputs():
    ''' initializes output filenames for all samples '''
    if VERBOSE:
        print('initializing outputs')
        print('--------------------')
    return [init_output(k) for k in range(NS)]


def init_header(k, output):
    ''' writes header for a sample '''
    # extract pressure/temperature indices from index
    i, j = np.unravel_index(k, shape=(NH, NT), order='C')
    with open(output[0], 'w') as dat_out:
        dat_out.write('# ---------------------\n')
        dat_out.write('# simulation parameters\n')
        dat_out.write('# ---------------------\n')
        dat_out.write('# nsmpl:    %d\n' % NSMPL)
        dat_out.write('# cutoff:   %d\n' % CUTOFF)
        dat_out.write('# mod:      %d\n' % MOD)
        dat_out.write('# nswps:    %d\n' % NSWPS)
        dat_out.write('# seed:     %d\n' % SEED)
        dat_out.write('# ---------------------\n')
        dat_out.write('# material properties\n')
        dat_out.write('# ---------------------\n')
        dat_out.write('# sites:    %d\n' % N)
        dat_out.write('# slices:   %d\n' % P)
        dat_out.write('# field:    %f\n' % H[i])
        dat_out.write('# temp:     %f\n' % T[j])
        dat_out.write('# inter:    %f\n' % J)
        dat_out.write('# jxy:      %f\n' % JXY[i, j])
        dat_out.write('# jz:       %f\n' % JZ[i, j])
        dat_out.write('# --------------------\n')
        dat_out.write('# | ener | mag | acc |\n')
        dat_out.write('# --------------------\n')


def init_headers():
    ''' writes headers for all samples '''
    if DASK:
        operations = [delayed(init_header)(k, OUTPUT[k]) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing headers')
            print('--------------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(init_header)(k, OUTPUT[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('initializing headers')
            print('--------------------')
            for k in tqdm(range(NS)):
                init_header(k, OUTPUT[k])
        else:
            for k in range(NS):
                init_header(k, OUTPUT[k])


def write_dat(output, state):
    ''' writes properties to dat file '''
    dat = output[0]
    ener, mag, acc = state[1:4]
    with open(dat, 'a') as dat_out:
        dat_out.write('%.4E %.4E %.4E\n' % (ener/(N*N*P), mag/(N*N*P), acc))


def write_dmp(output, state):
    ''' writes configurations to dmp file '''
    dmp = output[1]
    config = state[0]
    with open(dmp, 'ab') as dmp_out:
        np.savetxt(dmp_out, config.reshape(1, -1), fmt='%d')


def write_output(output, state):
    ''' writes output for a sample '''
    write_dat(output, state)
    write_dmp(output, state)


def write_outputs():
    ''' writes outputs for all samples '''
    if DASK:
        operations = [delayed(write_output)(OUTPUT[k], STATE[k]) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('\n---------------')
            print('writing outputs')
            print('---------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(write_output)(OUTPUT[k], STATE[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('writing outputs')
            print('---------------')
            for k in tqdm(range(NS)):
                write_output(OUTPUT[k], STATE[k])
        else:
            for k in range(NS):
                write_output(OUTPUT[k], STATE[k])


def consolidate_outputs():
    ''' consolidates outputs across samples '''
    if VERBOSE:
        print('---------------------')
        print('consolidating outputs')
        print('---------------------')
    dat = [OUTPUT[k][0] for k in range(NS)]
    dmp = [OUTPUT[k][1] for k in range(NS)]
    with open(file_prefix()+'.dat', 'w') as dat_out:
        for i in range(NH):
            for j in range(NT):
                k = np.ravel_multi_index((i, j), (NH, NT), order='C')
                with open(dat[k], 'r') as dat_in:
                    for line in dat_in:
                        dat_out.write(line)
    with open(file_prefix()+'.dmp', 'w') as dmp_out:
        for i in range(NH):
            for j in range(NT):
                k = np.ravel_multi_index((i, j), (NH, NT), order='C')
                with open(dmp[k], 'r') as dmp_in:
                    for line in dmp_in:
                        dmp_out.write(line)
    if VERBOSE:
        print('cleaning files')
        print('--------------')
    for k in range(NS):
        os.remove(dat[k])
        os.remove(dmp[k])

# ------------------------------------------------
# sample initialization and information extraction
# ------------------------------------------------


@nb.jit
def unravel(k, n):
    return k//n, k%n


@nb.jit
def ravel(i, j, n):
    return i*n+j


@nb.jit
def extract(config, k):
    ''' calculates the magentization and energy of a configuration '''
    i, j = unravel(k, NT)
    # magnetization
    mag = np.sum(config)
    ener = 0
    # loop through lattice
    for u in range(N):
        for v in range(N):
            for w in range(P):
                s = config[u, v, w]
                nn = JXY[i, j]*(config[(u+1)%N, v, w]+config[u, (v+1)%N, w])+JZ[i, j]*config[u, v, (w+1)%P]
                ener += s*nn
    ener *= -1
    return ener, mag


def init_sample(k):
    ''' initializes sample '''
    # generate random ising configuration
    config = np.random.choice([-1, 1], size=(N, N, P)).astype(np.int8)
    # extract energies and magnetizations
    ener, mag = extract(config, k)
    # set acceptations
    acc = 0.0
    # return configuration
    return [config, ener, mag, acc]


def init_samples():
    ''' initializes all samples '''
    if DASK:
        operations = [delayed(init_sample)(k) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing samples')
            print('--------------------')
            progress(futures)
            print('\n')
    elif PARALLEL:
        operations = [delayed(init_sample)(k) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('initializing samples')
            print('--------------------')
            futures = [init_sample(k) for k in tqdm(range(NS))]
        else:
            futures = [init_sample(k) for k in range(NS)]
    return futures

# ----------------
# monte carlo move
# ----------------


def spin_flip_mc(config, k, c, nts, nas):
    ''' spin flip monte carlo '''
    i, j = np.unravel_index(k, (NH, NT), order='C')
    nts += 1
    ener, _ = extract(config, k)
    u, v, w = c
    s = config[u, v, w]
    nn = JXY[i, j]*(config[(u-1)%N, v, w]+config[(u+1)%N, v, w]+config[u, (v-1)%N, w]+config[u, (v+1)%N, w])+\
         JZ[i, j]*(config[u, v, (w-1)%P]+config[u, v, (w+1)%P])
    de = 2*s*nn
    if de < 0 or np.random.rand() < np.exp(-de/T[j]):
        # update acceptations
        nas += 1
        config[u, v] *= -1
    # return spins and tries/acceptations
    return config, nts, nas


def total_spin_flip_mc(config, k):
    ''' spin flip monte carlo '''
    i, j = np.unravel_index(k, (NH, NT), order='C')
    ener, _ = extract(config, k)
    nener, _ = extract(-1*config, k)
    de = nener-ener
    if de < 0 or np.random.rand() < np.exp(-de/T[j]):
        config *= -1
    # return spins
    return config

# ---------------------
# monte carlo procedure
# ---------------------


def gen_sample(k, state):
    ''' generates a monte carlo sample '''
    config = state[0]
    nts, nas = 0., 0.
    # loop through monte carlo moves
    cs = np.random.permutation(list(it.product(np.arange(N), np.arange(N), np.arange(P))))
    for c in cs:
        config, nts, nas = spin_flip_mc(config, k, c, nts, nas)
    if np.random.rand() < 0.5:
        config = total_spin_flip_mc(config, k)
    # extract system properties
    ener, mag = extract(config, k)
    # acceptation ratio
    acc = nas/nts
    # return state
    return [config, ener, mag, acc]


def gen_samples():
    ''' generates all monte carlo samples '''
    if DASK:
        # list of delayed operations
        operations = [delayed(gen_sample)(k, STATE[k]) for k in range(NS)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('----------------------')
            print('performing monte carlo')
            print('----------------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(gen_sample)(k, STATE[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        # loop through pressures
        if VERBOSE:
            print('----------------------')
            print('performing monte carlo')
            print('----------------------')
            futures = [gen_sample(k, STATE[k]) for k in tqdm(range(NS))]
        else:
            futures = [gen_sample(k, STATE[k]) for k in range(NS)]
    return futures


# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------


def replica_exchange():
    ''' performs parallel tempering across temperature samples for each field strength '''
    # catalog swaps
    swaps = 0
    # loop through fields
    for u in range(NH):
        # loop through reference temperatures from high to low
        for v in range(NT-1, -1, -1):
            # loop through temperatures from low to current reference temperature
            for w in range(v):
                # extract index from each field/temperature index pair
                i = np.ravel_multi_index((u, v), (NH, NT), mode='raise', order='C')
                j = np.ravel_multi_index((u, w), (NH, NT), mode='raise', order='C')
                # calculate energy difference
                de = STATE[i][1]-STATE[j][1]
                # enthalpy difference
                dh = de*(1./T[v]-1./T[w])
                # metropolis criterion
                if np.random.rand() <= np.exp(dh):
                    swaps += 1
                    # swap states
                    STATE[j], STATE[i] = STATE[i], STATE[j]
    if VERBOSE:
        if PARALLEL:
            print('\n-------------------------------')
        print('%d replica exchanges performed' % swaps)
        print('-------------------------------')

# -------------
# restart files
# -------------


def load_samples_restart():
    ''' initialize samples with restart file '''
    if VERBOSE:
        if PARALLEL:
            print('\n----------------------------------')
        print('loading samples from previous dump')
        print('----------------------------------')
    return list(np.load(os.getcwd()+'/%s.%d.%d.rstrt.%d.npy' % (RENAME, N, P, RESTEP), allow_pickle=True))


def dump_samples_restart():
    ''' save restart state '''
    if VERBOSE:
        if PARALLEL:
            print('\n---------------')
        print('dumping samples')
    np.save(os.getcwd()+'/%s.%d.%d.rstrt.%d.npy' % (NAME, N, P, STEP+1), STATE)

# ----
# main
# ----

if __name__ == '__main__':

    (VERBOSE, PARALLEL, DASK, DISTRIBUTED, RESTART,
     REFREQ, RENAME, RESTEP,
     QUEUE, ALLOC, NODES, PPN,
     WALLTIME, MEM,
     NWORKER, NTHREAD, MTHD,
     NAME, N, P,
     J,
     NH, LH, HH,
     NT, LT, HT,
     CUTOFF, NSMPL, RECUTOFF) = parse_args()

    # set random seed
    SEED = 256
    np.random.seed(SEED)
    # processing or threading
    PROC = (NWORKER != 1)
    # ensure all flags are consistent
    if DISTRIBUTED and not DASK:
        DASK = 1
    if DASK and not PARALLEL:
        PARALLEL = 1

    # number of spinflips per sweep
    MOD = N*P
    # number of simulations
    NS = NH*NT
    # total number of monte carlo sweeps
    NSWPS = NSMPL*MOD

    # external field
    H = np.linspace(LH, HH, NH, dtype=np.float32)
    # temperature
    T = np.linspace(LT, HT, NT, dtype=np.float32)

    # interactions
    JXY = J/P*np.ones((NH, NT))
    JZ = -0.5*T.reshape(1, -1)*np.log(H.reshape(-1, 1)/(T.reshape(1, -1)*P))

    # dump external fields and temperatures
    np.save('{:s}.{:d}.{:d}.h.npy'.format(NAME, N, P), H)
    np.save('{:s}.{:d}.{:d}.t.npy'.format(NAME, N, P), T)
    np.save('{:s}.{:d}.{:d}.jxy.npy'.format(NAME, N, P), JXY)
    np.save('{:s}.{:d}.{:d}.jz.npy'.format(NAME, N, P), JZ)

    # -----------------
    # initialize client
    # -----------------

    if PARALLEL:
        from multiprocessing import freeze_support
    if not DASK:
        from joblib import Parallel, delayed
    if DASK:
        os.environ['DASK_ALLOWED_FAILURES'] = '64'
        os.environ['DASK_WORK_STEALING'] = 'True'
        os.environ['DASK_MULTIPROCESSING_METHOD'] = MTHD
        os.environ['DASK_LOG_FORMAT'] = '\r%(name)s - %(levelname)s - %(message)s'
        from distributed import Client, LocalCluster, progress
        from dask import delayed
    if DISTRIBUTED:
        from dask_jobqueue import PBSCluster

    if PARALLEL:
        freeze_support()
        if DASK and not DISTRIBUTED:
            # construct local cluster
            CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD, processes=PROC)
            # start client with local cluster
            CLIENT = Client(CLUSTER)
            # display client information
            if VERBOSE:
                client_info()
        if DASK and DISTRIBUTED:
            # construct distributed cluster
            CLUSTER = PBSCluster(queue=QUEUE, project=ALLOC,
                                 resource_spec='nodes=%d:ppn=%d' % (NODES, PPN),
                                 walltime='%d:00:00' % WALLTIME,
                                 processes=NWORKER, cores=NTHREAD*NWORKER, memory=str(MEM)+'GB',
                                 local_dir=os.getcwd())
            CLUSTER.start_workers(1)
            # start client with distributed cluster
            CLIENT = Client(CLUSTER)
            while 'processes=0 cores=0' in str(CLIENT.scheduler_info):
                time.sleep(5)
                if VERBOSE:
                    client_info()

    # -----------
    # monte carlo
    # -----------

    # define output file names
    OUTPUT = init_outputs()
    if CUTOFF < NSMPL:
        init_headers()
    # initialize simulation
    if RESTART:
        STATE = load_samples_restart()
        replica_exchange()
    else:
        if DASK:
            STATE = CLIENT.gather(init_samples())
        else:
            STATE = init_samples()
    # loop through to number of samples that need to be collected
    for STEP in tqdm(range(NSMPL)):
        if VERBOSE and DASK:
            client_info()
        # generate samples
        STATE[:] = gen_samples()
        # generate mc parameters
        if (STEP+1) > CUTOFF:
            # write data
            write_outputs()
        if DASK:
            # gather results from cluster
            STATE[:] = CLIENT.gather(STATE)
        if (STEP+1) % REFREQ == 0:
            # save state for restart
            dump_samples_restart()
        # replica exchange markov chain mc
        if (STEP+1) != NSMPL and (STEP+1) <= RECUTOFF:
            replica_exchange()
    if DASK:
        # terminate client after completion
        CLIENT.close()
    # consolidate output files
    if CUTOFF < NSMPL:
        consolidate_outputs()
