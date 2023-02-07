import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
cv_b_I, cv_tau_I = 0.5, 0.5

d_IE = 3.8*10**(-4)
T_I = 2

infection_type = 'prim' #'prim' or 'sec'
sim_kind = "agent"

runs = 10

def comp_MI(psi_N_I = 0.5, psi_N_c = 0.5, psi_cM_I = 0.5, psi_cM_c = 0.5, psi_E_I = 0.5, psi_E_c = 0.5):
    
    Is = np.random.lognormal(mean = np.log(np.array([b_I, tau_I])/np.sqrt(1 + np.array([cv_b_I, cv_tau_I])**2)), 
                            sigma = np.sqrt(np.log(1 + np.array([cv_b_I, cv_tau_I])**2)), size = (runs,2))

    # choose which parameter to vary
    run_data = np.array(Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(Is)/40),1))(delayed(sum_sim)(b_I = param[0], tau_I = param[1],
                                                                                                        regulation_coeffs = [psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c],
                                                                                                        infection = infection_type,
                                                                                                        sim_kind = sim_kind)
                                                    for param in Is))
        
    MI_args = np.arange(10,len(stat_names))
        
    MI_data = np.zeros(2*len(MI_args))
    for i in np.arange(len(MI_args)):
        MI_data[i] = calc_MI(run_data[:,8], run_data[:,MI_args[i]]) # 6 = a1_0, 7 = b_I
        MI_data[i + len(MI_args)] = calc_MI(run_data[:,7], run_data[:,MI_args[i]])
        
    return  np.array(np.append(MI_data, np.mean(run_data[:,10:], axis = 0)))


# function to slice data by parameters
extract_stats = np.vectorize(lambda x,stat: x[stat])

v_comp_MI = np.vectorize(comp_MI)

# functions for generating sobol sequence grids
def sample_grid(d,m, type = 'discrete'):
    
    if type == 'discrete':
        sample = np.array(list(itertools.product(np.arange(0,3)/2, repeat=d)))
    else:
        sampler = qmc.Sobol(d=d, scramble=False)
        vertices = np.array(list(itertools.product(np.arange(0,3)/2, repeat=d)))
        sample = np.vstack((sampler.random_base2(m=m), vertices))
    
    return sample

### (2) Maximize MI over $\psi_N^{(I)}$ and $\psi_N^{(c)}$

# Create grid of regulation coeffs
m = 0
regs_N = sample_grid(d=2,m=m, type = 'cts')

# Run simulations
MI_N_grid = np.array([comp_MI(psi_N_I = regs_N[i,0], psi_N_c = regs_N[i,1]) \
                      for i in np.arange(regs_N.shape[0])])

#np.save("_sim_data/"+isim_kind+'-'+infection_type+'-opt-N.npy', MI_N_grid)