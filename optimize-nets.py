import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed
import os

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import qmc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools

from stoch_sim_model import *

plt.style.use('custom.mplstyle')
%config InlineBackend.figure_format = 'retina'

### (1) Define function to run simulations and compute MI

# Make grid
b_I = 8
tau_I = 10
d_IE = 3.8*10**(-4)
T_I = 2

cv_b_I, cv_tau_I = 0.5, 0.5

infection_type = 'prim' #'prim' or 'sec'
sim_kind = "agent"

runs = 10

def run(regs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):
    
    Is = np.random.lognormal(mean = np.log(np.array([b_I, tau_I])/np.sqrt(1 + np.array([cv_b_I, cv_tau_I])**2)), 
                            sigma = np.sqrt(np.log(1 + np.array([cv_b_I, cv_tau_I])**2)), size = (runs,2))

    # choose which parameter to vary
    run_data = np.array(Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(Is)/40),1))(delayed(sum_sim)(b_I = param[0], tau_I = param[1],
                                                                                                        regulation_coeffs = regs,
                                                                                                        infection = infection_type,
                                                                                                        sim_kind = sim_kind)
                                                    for param in Is))
        
    MI_args = np.arange(10,len(stat_names))
        
    MI_data = np.zeros(2*len(MI_args))
    for i in np.arange(len(MI_args)):
        MI_data[i] = calc_MI(run_data[:,8], run_data[:,MI_args[i]]) # 6 = a1_0, 7 = b_I
        MI_data[i + len(MI_args)] = calc_MI(run_data[:,7], run_data[:,MI_args[i]])
        
    sim_data = np.array(np.append(MI_data, np.mean(run_data[:,10:], axis = 0)))
    
    out = np.concatenate((regs, sim_data), axis = None)
        
    return  out

# functions for generating sobol sequence grids
def sample_grid(d,m, type = 'discrete'):
    
    if type == 'discrete':
        sample = np.array(list(itertools.product(np.arange(0,3)/2, repeat=d)))
    else:
        sampler = qmc.Sobol(d=d, scramble=False)
        vertices = np.array(list(itertools.product(np.arange(0,3)/2, repeat=d)))
        if m > 0:
            sample = np.vstack((sampler.random_base2(m=m), vertices))
        else:
            sample = vertices
    
    return sample

### (2) Maximize MI over $\psi_N^{(I)}$ and $\psi_N^{(c)}$

# Create grid of regulation coeffs
m = 0
regs_all = sample_grid(d=6,m=m)

# Run simulations
MI_reg_grid = run(regs_all[0,:])

#np.save("_sim_data/"+isim_kind+'-'+infection_type+'-optnets-all.npy', MI_data)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--regulation_coeffs', dest='regs', type=float, required=True,
                        help='regulatory weights of the network')
    parser.add_argument('--outdir', dest='outdir', type=str, required=False, default=None,
                        help='/PATH/TO/WHERE/OUTPUT/IS/SAVED')
    parser.add_argument('--parallel', dest='parallel', action='store_true', required=False,
                        help='Enable parallel processing.')

    args = parser.parse_args()

    os.chdir(args.outdir)

    if args.parallel:
        Parallel(n_jobs=-1)(delayed(
            run)(args.regs, args.mutation_rate,
                args.off_rate, args.on_rate, child_rngs[i],
                intersample_times, num_samples, sample_sizes,
                i, args.outdir)
            for i in range(args.num_realizations))
    else:
        for i in range(args.num_realizations):
            run(pop_sizes, args.sequence_length, args.mutation_rate,
                args.off_rate, args.on_rate, child_rngs[i],
                intersample_times, num_samples, sample_sizes,
                i, args.outdir)


if __name__ == '__main__':
    main()