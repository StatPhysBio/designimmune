import os

from joblib import Parallel, delayed
import numpy as np
import pickle
import time

from stoch_sim_model import *

### Define function to run simulations and compute MI
num_cpu = 25 # number of CPUs used to benchmark expected runtime
add_cpu = 14 # additional cpus requested as a buffer
run_time = 4.0

def run(batch = 0, outdir='', comment = "sparse-reg", inf_sample = vir_prop_select, runs = 1, infection_type = 'prim', vir_model = "indep_harm", default_reg = act_psis + NE_psis + EM_psis + exp_psis, num_cpu = num_cpu, run_time = run_time):

    start = time.time() # timestamp start
    
    # Run simulations over different infections
    if "auto" in comment:
        inf_sample = np.array(np.meshgrid(d_S*np.array([1.0, 5.0]), # vary d_I
                                   S_0*np.array([1.0, 5.0]), # vary K_IE
                                   b_I*np.array([0.0]), # vary b_I
                                   K_EH*np.array([1.0]), # vary K_EH
                                   N_0*np.logspace(0.0, 0.0, 1) # vary N_0
                                           )).T.reshape(-1,5)
    
    batch_num = int((2400/len(inf_sample))*(run_time*num_cpu)/runs) + 1 # number of simulations to run on a cpu w/ max(#cpu) = 40 given virus conditions. # need later: 267906
    outfile = ('sim_batch_'+f'{batch[0]}-{runs}-{infection_type}-'+ f'{comment}.pkl')

    # Find psis to run
    if "full-reg" in comment:
        index_start, index_end = batch[0]*int(batch_num)/len(psi_2d)**3, (batch[0]+1)*int(batch_num)/len(psi_2d)**3 # fix psi for activation
        big_psis = np.array(list(itertools.product(psi_2d[int(index_start):int(index_end)+1].tolist(), psi_2d.tolist(), psi_2d.tolist(), psi_2d.tolist()))).reshape(-1,16)
        run_psis = big_psis[int(np.ceil((index_start - int(index_start))*len(psi_2d)**3)): int((index_end - int(index_start))*len(psi_2d)**3)]

    elif "full_nobias-reg" in comment:
        index_start, index_end = batch[0]*int(batch_num)/len(psi_2d_nobias)**3, (batch[0]+1)*int(batch_num)/len(psi_2d_nobias)**3 # fix psi for activation
        big_psis = np.array(list(itertools.product(psi_2d_nobias[int(index_start):int(index_end)+1].tolist(), psi_2d_nobias.tolist(), psi_2d_nobias.tolist(), psi_2d_nobias.tolist()))).reshape(-1,16)
        run_psis = big_psis[int(np.ceil((index_start - int(index_start))*len(psi_2d_nobias)**3)): int((index_end - int(index_start))*len(psi_2d_nobias)**3)]
        
    elif "sparse-reg" in comment:
        index_start, index_end =int(batch[0]*batch_num), int((batch[0]+1)*batch_num)
        run_psis = psi_2d_sparse[index_start:index_end]
        
    elif "single-reg" in comment:
        run_psis = np.array([default_reg])
        
    else:
        index_start, index_end =int(batch[0]*batch_num), int((batch[0]+1)*batch_num)
        run_psis = np.array(list(itertools.product(psi_2d.tolist() if ("Nact" in comment and "comp_bias" not in comment and "auto" not in comment) else (psi_2d_comp_bias.tolist() if "comp_bias" in comment else (psi_2d.tolist() if "auto" in comment else [[psi_max, psi_max, 0.0, -F0_max], [psi_max, psi_max, 0.0, 0.0], [psi_max, psi_max, 0.0, F0_max]])),
                                       psi_2d.tolist() if ("NE" in comment and "auto" not in comment) else ([[0.0, 0.0, 0.0, -F0_max]] if "auto" in comment else [[0.0, 0.0, 0.0, -F0_max], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, F0_max]]), 
                                       psi_2d.tolist() if ("EM" in comment and "auto" not in comment) else ([[0.0, 0.0, 0.0, -F0_max]] if "auto" in comment else [[0.0, 0.0, 0.0, -F0_max], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, F0_max]]), 
                                       psi_2d.tolist() if ("Ediv" in comment and "comp_bias" not in comment and "auto" not in comment) else (psi_2d_comp_bias.tolist() if "comp_bias" in comment else (psi_2d.tolist() if "auto" in comment else [[psi_max, psi_max, 0.0, -F0_max], [psi_max, psi_max, 0.0, 0.0], [psi_max, psi_max, 0.0, F0_max]]))))).reshape(-1,16)[index_start:index_end]

    params = [np.concatenate(q) for q in list(itertools.product( np.tile(inf_sample, (runs,1)).tolist(), run_psis.tolist()))]
    
    print(f'Running {len(params)} simulations in batch #{batch[0]}')

    psi_list = Parallel(n_jobs = num_cpu + add_cpu, batch_size = max(int(len(params)/(num_cpu + add_cpu)),1))(delayed(lin_stoch_sim)(d_I = param[0], 
                                                                                                               K_IE = param[1],
                                                                                                               b_I = param[2],
                                                                                                               K_EH = param[3],
                                                                                                               N_0 = param[4],
                                                                                                               activation_regulation = param[5:9],
                                                                                                               NE_regulation = param[9:13],
                                                                                                               EM_regulation = param[13:17],
                                                                                                               expansion_regulation = param[17:21],
                                                                                                               infection = infection_type,
                                                                                                               vir_model = vir_model if 'auto' not in comment else "autoimmune",
                                                                                                               reg_model = "competition_model" if "comp_model" in comment else "mwc_like")
                                                    for param in params)

    # Store data in dictionary
    psi_dict = {}
    select_keys = ['parameters', 'summary_stats']
    for k in select_keys:
        psi_dict[k] = list(d[k] for d in psi_list)

    # Save dictionary
    with open(os.path.join(outdir, outfile), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(psi_dict, f, pickle.HIGHEST_PROTOCOL)
    print(f'Saved {outfile}')

    end = time.time() # timestamp end
    
    print((end - start)/3600, 'hrs')

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