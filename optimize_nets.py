import os

from joblib import Parallel, delayed
import numpy as np
import pickle

from stoch_sim_model import *

### Define function to run simulations and compute MI
num_cpu = 40 # number of CPUs requested
run_time = 4.0

def run(batch = 0, outdir='', comment = "Nact-Ediv-vir", inf_sample = vir_prop, runs = 1, infection_type = 'prim', vir_model = "indep_harm", default_reg = act_psis + NM_psis + EM_psis + exp_psis, num_cpu = num_cpu):
    
    # Run simulations over different infections
    if "auto" in comment:
        inf_sample = np.array([[d_S, K_SE, 0.0], [0.0, K_SE, 0.0]])
    
    batch_num = int((288000/len(inf_sample))*(run_time/4)/runs) + 1 # number of simulations to run on a cpu w/ max(#cpu) = 40 given virus conditions.
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
        
    elif "single-reg" in comment:
        run_psis = np.array([default_reg])
        
    else:
        index_start, index_end =int(batch[0]*batch_num), int((batch[0]+1)*batch_num)
        run_psis = np.array(list(itertools.product(psi_2d.tolist() if ("Nact" in comment and "comp_model" not in comment) else (psi_2d_comp.tolist() if "comp_model" in comment else [[psi_max/2, psi_max/2, 0.0, -2.0], [psi_max/2, psi_max/2, 0.0, 0.0], [psi_max/2, psi_max/2, 0.0, 2.0]]),
                                       psi_2d.tolist() if "NM" in comment else [[0.0, 0.0, 0.0, -2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]], 
                                       psi_2d.tolist() if "EM" in comment else [[0.0, 0.0, 0.0, -2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]], 
                                       psi_2d.tolist() if ("Ediv" in comment and "comp_model" not in comment) else (psi_2d_comp.tolist() if "comp_model" in comment else [[psi_max/2, psi_max/2, 0.0, -2.0], [psi_max/2, psi_max/2, 0.0, 0.0], [psi_max/2, psi_max/2, 0.0, 2.0]])))).reshape(-1,16)[index_start:index_end]

    params = [np.concatenate(q) for q in list(itertools.product( np.tile(inf_sample, (runs,1)).tolist(), run_psis.tolist()))]
    
    print(f'Running {len(params)} simulations in batch #{batch[0]}')

    psi_list = Parallel(n_jobs = num_cpu, batch_size = max(int(len(inf_sample)/num_cpu),1))(delayed(lin_stoch_sim)(d_I = param[0], 
                                                                                                               K_IE = param[1],
                                                                                                               b_I = param[2],
                                                                                                               K_EI = param[1],
                                                                                                               K_EH = K_EH,
                                                                                                               activation_regulation = param[3:7], # if "Nact-reg" in comment else act_psis,
                                                                                                               NM_regulation = param[7:11], # if "NM-reg" in comment else NM_psis,
                                                                                                               EM_regulation = param[11:15], # if "EM-reg" in comment else EM_psis,
                                                                                                               expansion_regulation = param[15:19], # if "Ediv-reg" in comment else act_psis,
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