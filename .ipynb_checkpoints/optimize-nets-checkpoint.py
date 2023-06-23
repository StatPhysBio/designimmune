import os

from joblib import Parallel, delayed
import numpy as np

from stoch_sim_model import *

### (1) Define function to run simulations and compute MI

# Make grid
infection_type = 'prim' # 'prim' or 'sec'
sim_kind = "agent"
reg_model = "mwc_like" # "mwc_like", "hill_and", "hill_or"
comment = "no-cell-var"

S_0 = 10_000_000 #susceptible cells
d_S = 0.05
I_0 = 10 # initial detectable levelof infected cells
#b_I = 1*(10**(-6)) # harm per unit virion
N_0 = 50
K_IE = 7.8*(10**3)

runs = 500
reg_weight = 1

def run(regs = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0], outdir=''):
    
    reg_coeff = reg_weight*(np.array(regs) - 1)
    
    print(f'Running with regs = {list(reg_coeff)}')
    
    # sample distribution of pathogen killing rate and size of naive repertoire
    dIs = sample_grid(d = 1, l_bounds = d_S, u_bounds = 1.0, runs = runs)

    # choose which parameter to vary
    print('Running simulation')
    run_data = np.vstack(Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(dIs)/20),1))(delayed(sum_sim)(d_I = param, 
                                                                                                               N_0 = N_0,
                                                                                                               K_IE = K_IE,
                                                                                                               regulation_coeffs = reg_coeff,
                                                                                                               infection = infection_type,
                                                                                                               vir_model = 'dep_harm',
                                                                                                               sim_kind = sim_kind,
                                                                                                               reg_model = reg_model)
                                                    for param in dIs ))

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