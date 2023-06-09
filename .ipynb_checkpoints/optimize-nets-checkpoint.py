import os

from joblib import Parallel, delayed
import numpy as np

from stoch_sim_model import *

### (1) Define function to run simulations and compute MI

# Make grid
infection_type = 'sec' # 'prim' or 'sec'
sim_kind = "agent"
reg_model = "mwc_like" # "mwc_like", "hill_and", "hill_or"
comment = "no-cell-var"

S_0 = 10_000_000 #susceptible cells
d_S = 0.1
b_S = S_0*d_S
I_0 = 10 # initial detectable levelof infected cells
b_I = 1*(10**(-6)) # harm per unit virion
N_0 = 50
K_IE = 7.8*(10**3)

runs = 100
reg_weight = 1

def run(regs = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0], outdir=''):
    
    reg_coeff = reg_weight*(np.array(regs) - 1)
    
    print(f'Running with regs = {list(reg_coeff)}')
    
    # sample distribution of pathogen killing rate and size of naive repertoire
    dI_N0s = sample_2d(l_bounds = [0, N_0-0.5], u_bounds = [1.25, N_0], runs = runs)

    # choose which parameter to vary
    print('Running simulation')
    run_data = np.array(Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(dI_N0s)/20),1))(delayed(sum_sim)(d_I = params[0], N_0 = N_0,
                                                                                                                  K_IE = K_IE,
                                                                                                               regulation_coeffs = reg_coeff,
                                                                                                               infection = infection_type,
                                                                                                               sim_kind = sim_kind,
                                                                                                               reg_model = reg_model)
                                                    for params in dI_N0s ))
        
#     print('Computing MI')
#     MI_args = np.arange(10,len(stat_names))
        
#     MI_data = np.zeros(2*len(MI_args))
#     for i in np.arange(len(MI_args)):
#         MI_data[i] = calc_MI(run_data[:,8], run_data[:,MI_args[i]]) # 8 = N_0, 7 = d_I
#         MI_data[i + len(MI_args)] = calc_MI(run_data[:,7], run_data[:,MI_args[i]])
        
#     # Combined dataset with [regs, MI-N_0, MI-d_I, mean-response] 
#     sim_data = np.array(np.append(MI_data, np.mean(run_data[:,10:], axis = 0)))
    
#     print('Concatenting')
#     out = np.concatenate((reg_weight*(np.array(regs) - 1), sim_data), axis = None)

    outfile = '-'.join((regs * 1).astype('U1')) + f'-{runs}-{infection_type}-{reg_model}-{comment}.npy'
    
    outfile = os.path.join(outdir, outfile) # what is outfile?
    print('Saving')
    np.save(outfile, run_data)
    print(f'Saved {outfile}')

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