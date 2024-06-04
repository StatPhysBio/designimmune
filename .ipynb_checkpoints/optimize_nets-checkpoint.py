import os

from joblib import Parallel, delayed
import numpy as np
import pickle

from stoch_sim_model import *

### Define function to run simulations and compute MI

# Set simulation parameters
runs = 1
vir_samp = np.tile(vir_prop, (runs,1)) # sample distribution of pathogen killing rate and size of naive repertoire
vir_choice = 0
num_cpu = 40 # number of CPUs requested
batch_num = int(62500/len(vir_samp)) + 1 # number of network variants to run on a cpu

def run(batch = 0, outdir='', comment = "Nact-Ediv-vir", virus_sample = vir_samp, infection_type = 'prim', vir_model = "indep_harm", default_reg = act_psis + NM_psis + EM_psis + exp_psis):
    
    # Run simulations over different infections
    print(f'Running simulations in batch #{batch[0]}')
    outfile = ('sim_batch_'+f'{batch[0]}-{runs}-{infection_type}-'+ f'{comment}.pkl')

    # Find psis to run
    if "full-reg" in comment:
        index_start, index_end = batch[0]*int(batch_num)/len(psi_2d)**3, (batch[0]+1)*int(batch_num)/len(psi_2d)**3
        big_psis = np.array(list(itertools.product(psi_2d[int(index_start):int(index_end)+1].tolist(), psi_2d.tolist(), psi_2d.tolist(), psi_2d.tolist()))).reshape(-1,12)
        run_psis = big_psis[int(np.ceil((index_start - int(index_start))*len(psi_2d)**3)): int((index_end - int(index_start))*len(psi_2d)**3)]
        
    elif "single-reg" in comment:
        run_psis = np.array(default_reg)
        
    else:
        run_psis = np.array(list(itertools.product(psi_2d.tolist() if "Nact" in comment else [act_psis], 
                                       psi_2d.tolist() if "NM" in comment else [NM_psis], 
                                       psi_2d.tolist() if "EM" in comment else [EM_psis], 
                                       psi_2d.tolist() if "Ediv" in comment else [exp_psis]))).reshape(-1,12)

    params = [np.concatenate(q) for q in list(itertools.product(virus_sample.tolist(), run_psis.tolist()))]

    psi_list = Parallel(n_jobs = num_cpu, batch_size = max(int(len(virus_sample)/num_cpu),1))(delayed(lin_stoch_sim)(d_I = param[0], 
                                                                                                               K_IE = param[1],
                                                                                                               b_I = param[2],
                                                                                                               K_EI = param[1],
                                                                                                               K_EH = K_EH,
                                                                                                               activation_regulation = param[3:6], # if "Nact-reg" in comment else act_psis,
                                                                                                               NM_regulation = param[6:9], # if "NM-reg" in comment else NM_psis,
                                                                                                               EM_regulation = param[9:12], # if "EM-reg" in comment else EM_psis,
                                                                                                               expansion_regulation = param[12:], # if "Ediv-reg" in comment else act_psis,
                                                                                                               infection = infection_type,
                                                                                                               vir_model = vir_model)
                                                    for param in params)

    # Store data in dictionary
    psi_dict = {}
    select_keys = ['cell_time_series', 'prim_diff_bias', 'sec_diff_bias', 'parameters', 'summary_stats']
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