import os

from joblib import Parallel, delayed
import numpy as np
import pickle

from stoch_sim_model import *

### Define function to run simulations and compute MI

# Set simulation parameters
runs = 5
N_0 = 300
vir_samp = np.tile(vir_prop, (runs,1)) # sample distribution of pathogen killing rate and size of naive repertoire
reg_weight = 1
num_cpu = 10

def run(reg_opt = 0, outdir='', virus_sample = vir_samp, infection_type = 'sec', comment = "lin-based"):
    
    reg_coeff = reg_weight*psi_opts[int(reg_opt[0])]
    outfile = ('-'.join((reg_coeff).astype('U4'))
               + f'-{runs}-{infection_type}-'
               + f'-{comment}.pkl')
    outfile = os.path.join(outdir, outfile)
    
    # Run simulations over different infections
    print(f'Running simulation with regs = {list(reg_coeff)}')

    runs_list = Parallel(n_jobs=os.cpu_count(), batch_size = max(int(len(virus_sample)/num_cpu),1))(delayed(lin_stoch_sim)(d_I = param[0], 
                                                                                                               K_IE = param[1],
                                                                                                               K_EI = param[1]*param[2],
                                                                                                               K_EH = K_IH*param[3],
                                                                                                               regulation_coeffs = reg_coeff,
                                                                                                               infection = infection_type,
                                                                                                               vir_model = 'dep_harm')
                                                    for param in virus_sample)

    # Store data in dictionary
    runs_dict = {}
    select_keys = ['reg_coeffs','cell_time_series','time','prim_diff_bias', 'sec_diff_bias','lineage_diff', 'parameters', 'summary_stats']
    for k in select_keys:
      runs_dict[k] = list(d[k] for d in runs_list)

    with open(outfile, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(runs_dict, f, pickle.HIGHEST_PROTOCOL)
    print(f'Saved {outfile}')

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--reg_opt', dest='reg_opt', nargs='+', type=float, required=True,
                        help='regulatory weights of the network')
    parser.add_argument('--outdir', dest='outdir', type=str, required=False, default='',
                        help='/PATH/TO/WHERE/OUTPUT/IS/SAVED')

    args = parser.parse_args()

    os.chdir(args.outdir)
    
    run(args.reg_opt, args.outdir)


if __name__ == '__main__':
    main()