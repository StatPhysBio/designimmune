import os

from joblib import Parallel, delayed
import numpy as np

from stoch_sim_model import *

### (1) Define function to run simulations and compute MI

# Make grid
infection_type = 'sec' # 'prim' or 'sec'
sim_kind = "agent"
reg_model = "hill_or" # "mwc_like", "hill_and", "hill_or"
comment = "no-cell-var"

S_0 = 10_000_000 #susceptible cells
d_S = 0.05
I_0 = 10 # initial detectable levelof infected cells
#b_I = 1*(10**(-6)) # harm per unit virion
N_0 = 100
K_IE = 10**4
d_I = 10*d_S

runs = 100
reg_weight = 1

def run(regs = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0], reg_logs = np.array([0,0,0]), outdir=''):
    
    reg_coeff = reg_weight*(np.array(regs) - 1)
    
    print(f'Running with regs = {list(reg_coeff)}')
    print(f'Running with logic = {list(reg_logs)}')
    
    outfile = ('-'.join((regs * 1).astype('U1'))
               + f'-{runs}-{infection_type}-'
               + '-'.join((reg_logs * 1).astype('U1'))
               + f'-{comment}.npy')
    outfile = os.path.join(outdir, outfile) # what is outfile?
    print(f'Will save to {outfile}.')
    
    # sample distribution of pathogen killing rate and size of naive repertoire
    vir_prop = np.array(np.meshgrid(d_S*np.array([0, 2, 5, 10, 20]), K_IE*np.array([1, 2, 5, 10, 20]))).T.reshape(-1,2)
    dIs = np.tile(vir_prop, (int(runs/vir_prop.shape[0]),1))
    
    # choose which parameter to vary
    print('Running simulation')
    run_data = np.vstack(Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(dIs)/20),1))(delayed(sum_sim)(d_I = param[0], 
                                                                                                               N_0 = N_0,
                                                                                                               K_IE = param[1],
                                                                                                               regulation_coeffs = reg_coeff,
                                                                                                               infection = infection_type,
                                                                                                               vir_model = 'dep_harm',
                                                                                                               sim_kind = sim_kind,
                                                                                                               reg_logs = reg_logs)
                                                    for param in dIs ))

    print('Saving')
    np.save(outfile, run_data)
    print(f'Saved {outfile}')

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--reg_coefs', dest='regs', nargs='+', type=float, required=True,
                        help='regulatory weights of the network')
    parser.add_argument('--reg_logs', dest='reg_logs', nargs='+', type=int, required=True,
                        help='regulatory logic of the network')
    parser.add_argument('--outdir', dest='outdir', type=str, required=False, default='',
                        help='/PATH/TO/WHERE/OUTPUT/IS/SAVED')

    args = parser.parse_args()

    os.chdir(args.outdir)
    
    run(np.array(args.regs), np.array(args.reg_logs), args.outdir)


if __name__ == '__main__':
    main()