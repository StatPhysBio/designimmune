import os

from joblib import Parallel, delayed
import numpy as np
import pickle

from stoch_sim_model import *

### Define function to run simulations and compute MI

# Set simulation parameters
runs = 200
N_0 = 100
vir_samp = np.tile(vir_prop, (int(runs/vir_prop.shape[0]),1)) # sample distribution of pathogen killing rate and size of naive repertoire
reg_weight = 1

def run(regs = [2.0, 2.0, 0.0, 0.0, 2.0, 2.0], reg_logs = np.array([0,0,0]), outdir='', virus_sample = vir_samp, infection_type = 'sec', comment = "no-cell-var"):
    
    reg_coeff = reg_weight*(np.array(regs) - 1)
    
    print(f'Running with regs = {list(reg_coeff)}')
    print(f'Running with logic = {list(reg_logs)}')
    
    outfile = ('-'.join((regs * 1).astype('U1'))
               + f'-{runs}-{infection_type}-'
               + '-'.join((reg_logs * 1).astype('U1'))
               + f'-{comment}.pkl')
    outfile = os.path.join(outdir, outfile) # what is outfile?
    print(f'Will save to {outfile}.')
    
    # Run simulations over different infections
    print('Running simulation')
    runs_list = Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(virus_sample)/20),1))(delayed(agent_stoch_sim)(d_I = param[0], 
                                                                                                               K_IE = param[1],
                                                                                                               regulation_coeffs = reg_coeff,
                                                                                                               infection = infection_type,
                                                                                                               vir_model = 'dep_harm',
                                                                                                               reg_logs = reg_logs)
                                                    for param in virus_sample)

    # Store data in dictionary
    runs_dict = {}
    select_keys = ['reg_coeffs','cell_time_series','time','prim_diff_bias', 'sec_diff_bias','lineage_diff', 'parameters', 'sumary_stats']
    for k in select_keys:
      runs_dict[k] = list(d[k] for d in runs_list)

    print('Saving')

    with open(outfile, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(runs_dict, f, pickle.HIGHEST_PROTOCOL)
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