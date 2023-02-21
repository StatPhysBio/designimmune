import os

from joblib import Parallel, delayed
import numpy as np

from stoch_sim_model import *

### (1) Define function to run simulations and compute MI

# Make grid
b_I = 8
tau_I = 10
d_IE = 3.8*10**(-4)
T_I = 2

cv_b_I, cv_tau_I = 0.5, 0.5

infection_type = 'sec' #'prim' or 'sec'
sim_kind = "agent"

runs = 1000

def run(regs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], outdir=''):
    print(f'Running with regs = {list(regs)}')
    Is = np.random.lognormal(mean = np.log(np.array([b_I, tau_I])/np.sqrt(1 + np.array([cv_b_I, cv_tau_I])**2)), 
                            sigma = np.sqrt(np.log(1 + np.array([cv_b_I, cv_tau_I])**2)), size = (runs,2))

    # choose which parameter to vary
    print('Running simulation')
    run_data = np.array(Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(Is)/40),1))(delayed(sum_sim)(b_I = param[0], tau_I = param[1],
                                                                                                        regulation_coeffs = regs,
                                                                                                        infection = infection_type,
                                                                                                        sim_kind = sim_kind)
                                                    for param in Is))
        
    print('Computing MI')
    MI_args = np.arange(10,len(stat_names))
        
    MI_data = np.zeros(2*len(MI_args))
    for i in np.arange(len(MI_args)):
        MI_data[i] = calc_MI(run_data[:,8], run_data[:,MI_args[i]]) # 8 = tau_I, 7 = b_I
        MI_data[i + len(MI_args)] = calc_MI(run_data[:,7], run_data[:,MI_args[i]])
        
    sim_data = np.array(np.append(MI_data, np.mean(run_data[:,10:], axis = 0)))
    
    print('Concatenting')
    out = np.concatenate((regs, sim_data), axis = None)

    outfile = '-'.join((regs * 2).astype('U1')) + f'-nruns_{runs}-inftype_{infection_type}.npy'
    
    outfile = os.path.join(outdir, outfile) # what is outfile?
    print('Saving')
    np.save(outfile, out)
    print(f'Saved {outfile}')

### (2) Maximize MI over $\psi_N^{(I)}$ and $\psi_N^{(c)}$

# # Create grid of regulation coeffs
# m = 0
# regs_all = sample_grid(d=6,m=m)

# # Run simulations
#MI_reg_grid = run(regs_all[0,:])

#np.save("_sim_data/"+isim_kind+'-'+infection_type+'-optnets-all.npy', MI_data)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--reg_coefs', dest='regs', nargs='+', type=float, required=True,
                        help='regulatory weights of the network')
    parser.add_argument('--outdir', dest='outdir', type=str, required=False, default='',
                        help='/PATH/TO/WHERE/OUTPUT/IS/SAVED')

    args = parser.parse_args()

    os.chdir(args.outdir)
    
    run(np.array(args.regs), args.outdir)


if __name__ == '__main__':
    main()