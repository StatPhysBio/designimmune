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

from stoch_sim_model import *

plt.style.use('/Users/ObinnaUkogu/Desktop/custom.mplstyle')
%config InlineBackend.figure_format = 'retina'

### (1) Set pathogen parameters

# Make grid
b_I = 8
t1 = 10
cv_b_I, cv_t1 = 2, 0.5

d_IE = 3.8*10**(-4)
T_I = 2

infection_type = 'prim' #'prim'
sim_kind = "pop_ode"

runs = 1000

MI_args = np.hstack(([7],np.arange(10,len(stat_names))))

def comp_MI(psi_N_I = 0.5, psi_N_c = 0.5, psi_cM_I = 0.5, psi_cM_c = 0.5, psi_E_I = 0.5, psi_E_c = 0.5):
    
    Is = np.random.lognormal(mean = np.log(np.array([b_I, t1])/np.sqrt(1 + np.array([cv_b_I, cv_t1])**2)), 
                            sigma = np.sqrt(np.log(1 + np.array([cv_b_I, cv_t1])**2)), size = (runs,2))

    # choose which parameter to vary
    run_data = np.array(Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(Is)/8),1))(delayed(sum_sim)(b_I = param[0], t1 = param[1],
                                                                                                        regulation_coeffs = [psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c],
                                                                                                        infection = infection_type,
                                                                                                        sim_kind = sim_kind)
                                                    for param in Is))
    
    MI_data = np.zeros(len(MI_args))
    for i in np.arange(len(MI_data)):
        MI_data[i] = calc_MI(run_data[:,7], run_data[:,MI_args[i]]) # 6 = a1_0, 7 = b_I
        
    return  np.array(np.append(MI_data, np.mean(run_data[:,10])), dtype=object)


# function to slice data by parameters
extract_stats = np.vectorize(lambda x,stat: x[stat])

v_comp_MI = np.vectorize(comp_MI)

### (1) Compute MI over all networks

# Create grid of differentiation rates
def optimize_all_regs(grid_num = 4):
    
    # create grid
    psi_N_Is, psi_N_cs, psi_cM_Is, psi_cM_cs, psi_E_Is, psi_E_c = np.meshgrid(np.linspace(0.00, 1.0, grid_num),
                                                                              np.linspace(0.00, 1.0, grid_num),
                                                                              np.linspace(0.00, 1.0, grid_num),
                                                                              np.linspace(0.00, 1.0, grid_num),
                                                                              np.linspace(0.00, 1.0, grid_num),
                                                                              np.linspace(0.00, 1.0, grid_num))

    regs = (np.vstack((np.ravel(psi_N_Is), np.ravel(psi_N_cs), 
                       np.ravel(psi_cM_Is), np.ravel(psi_cM_cs),
                       np.ravel(psi_E_Is), np.ravel(psi_E_c))).T)

    # Run simulations
    MI_reg_grid = v_comp_MI(psi_N_I = regs[:,0], psi_N_c = regs[:,1], 
                          psi_cM_I = regs[:,2], psi_cM_c = regs[:,3], 
                          psi_E_I = regs[:,4], psi_E_c = regs[:,5])

    # Match networks with MI outputs
    MI_data = np.zeros((len(MI_reg_grid),len(MI_reg_grid[0])))

    for i in np.arange(len(MI_reg_grid)):
        MI_data[i,:] = MI_reg_grid[i].astype(float)

    raw_network_MI = np.append(regs, MI_data[:,1:], axis = 1)

    # Remove biologically infeasible network architectures
    keep_row = np.ones(raw_network_MI.shape[0], dtype = bool)
    for i in np.arange(raw_network_MI.shape[0]):
        if raw_network_MI[i,-1] == 0.0 or raw_network_MI[i,-1] == 0.0:
            keep_row[i] = False

    network_MI = (raw_network_MI[keep_row,...])
    MI_data_no_zero = (MI_data[keep_row,...])

    # Plot pareto front for appropriateness and effectiveness of the immune response
    fig_b_I = plt.figure(figsize=(12,12),dpi = 150)

    i = 1

    for val in stat_names[i+9:]:

        xy = np.array([np.nan_to_num(MI_data_no_zero[:,i], copy = False) ,np.nan_to_num(MI_data_no_zero[:,-1], copy = False)]).T

        # hull = ConvexHull(np.array([x_, y_]).T)

        ax = fig_b_I.add_subplot(4, 2, i)
        cs = ax.plot(xy[:,0], xy[:,1], 'ko')

        # find pareto front
        remove_row = np.ones(xy.shape[0], dtype = bool)

        for j in np.arange(len(xy[:,0])):
            for k in np.arange(len(xy[:,0])):
                if xy[j,0] < xy[k,0] and xy[j,1] > xy[k,1]:
                    remove_row[j] = False


        x_front = (xy[:,0])[remove_row,...]
        y_front = (xy[:,1])[remove_row,...]
        ax.plot(x_front, y_front, 'ro')

        # set the limits of the plot to the limits of the data
        ax.set(xlabel=r"$I($"+val+r";$b_I)$", ylabel=r"$\mathbb{E}_{\sim b_I}[$"+stat_names[10]+r"$]$")

        i += 1

    fig_b_I.suptitle(r'Pareto front over $(\psi_{N}^{(I)},\psi_{N}^{(c)},\psi_{cM}^{(I)},\psi_{cM}^{(c)}, \psi_{E}^{(I)}, \psi_{E}^{(c)})$'+' for'+infection_type+". infection")
    plt.tight_layout()

    plt.show()
    
    return fig_b_I