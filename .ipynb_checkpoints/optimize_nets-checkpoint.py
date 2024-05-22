import os

from joblib import Parallel, delayed
import numpy as np
import pickle

from stoch_sim_model import *

### Define function to run simulations and compute MI

# Set simulation parameters
runs = 1
vir_samp = np.tile(vir_prop, (runs,1)) # sample distribution of pathogen killing rate and size of naive repertoire
num_cpu = 40 # number of CPUs requested
batch_num = 4*19200 + 1 # number of network variants to run on a cpu

def run(batch = 0, outdir='', comment = "full-reg-vir-0", virus_sample = vir_samp[0], infection_type = 'prim'):
    
    # Run simulations over different infections
    print(f'Running simulations in batch #{batch[0]}')
    outfile = ('sim_batch_'+f'{batch[0]}-{runs}-{infection_type}-'+ f'{comment}.pkl')

    # Find psis to run
    index_start, index_end = batch[0]*batch_num/len(psi_2d)**3, (batch[0]+1)*batch_num/len(psi_2d)**3
    big_psis = np.array(list(itertools.product(psi_2d[int(index_start):int(index_end)+1].tolist(), psi_2d.tolist(), psi_2d.tolist(), psi_2d.tolist()))).reshape(-1,12)
    run_psis = big_psis[int(np.ceil((index_start - int(index_start))*len(psi_2d)**3)): int((index_end - int(index_start))*len(psi_2d)**3)]

    psi_list = Parallel(n_jobs = num_cpu, batch_size = max(int(len(virus_sample)/num_cpu),1))(delayed(lin_stoch_sim)(d_I = virus_sample[0], 
                                                                                                               K_IE = virus_sample[1],
                                                                                                               K_EI = virus_sample[1]*virus_sample[2],
                                                                                                               K_EH = K_EH*virus_sample[3],
                                                                                                               NM_regulation = regs[3:6], # if "NM-reg" in comment else NM_psis,
                                                                                                               EM_regulation = regs[6:9], # if "EM-reg" in comment else EM_psis,
                                                                                                               activation_regulation = regs[0:3], # if "Nact-reg" in comment else act_psis,
                                                                                                               expansion_regulation = regs[9:], # if "Ediv-reg" in comment else act_psis,
                                                                                                               infection = infection_type,
                                                                                                               vir_model = 'dep_harm')
                                                    for regs in run_psis)

    # Store data in dictionary
    psi_dict = {}
    select_keys = ['cell_time_series', 'prim_diff_bias', 'sec_diff_bias', 'parameters', 'summary_stats', 'pmemory_survived']
    for k in select_keys:
        psi_dict[k] = list(d[k] for d in psi_list)

    # Save dictionary
    with open(os.path.join(outdir, "raw",outfile), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(psi_dict, f, pickle.HIGHEST_PROTOCOL)
    print(f'Saved {outfile}')

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--batch', dest='batch', nargs='+', type=float, required=True,
                        help='batch of regulatory weights of the network')
    parser.add_argument('--outdir', dest='outdir', type=str, required=False, default='',
                        help='/PATH/TO/WHERE/OUTPUT/IS/SAVED')
    parser.add_argument('--comment', dest='comment', type=str, required=True, default='',
                        help='simulation specifications')

    args = parser.parse_args()

    os.chdir(args.outdir)
    
    run(args.batch, args.outdir)


if __name__ == '__main__':
    main()