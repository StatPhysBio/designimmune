import os
from joblib import Parallel, delayed
import numpy as np
import pickle
import time

from stoch_sim_model import *

### Define function to run simulations and compute MI
num_cpu = 25 # number of CPUs used to benchmark expected runtime
add_cpu = 15 # additional cpus requested as a buffer
run_time = 4.0

def run(
    batch = 0,
    outdir='',
    comment = "sparse-reg",
    inf_sample = infection_sample_select,
    runs = 1,
    infection_model = "acute_all", #'acute_all',
    default_reg = np.array([act_psis + NE_psis + EM_psis + contract_psis]),
    sim_per_cpu_hour = 1800,
    num_cpu = num_cpu,
    run_time = run_time,
    seed = None
):

    start = time.time() # timestamp start

    # Run simulations over different infections

    batch_num = int((sim_per_cpu_hour/len(inf_sample))*(run_time*num_cpu)/runs) + 1 # number of simulations to run on a cpu w/ max(#cpu) = 40 given virus conditions.
    outfile = ('sim_batch_'+f'{batch[0]}-{runs}-{infection_model}-'+ f'{comment}.pkl')

    # Find psis to run
    if "full-reg" in comment:
        index_start, index_end = batch[0]*int(batch_num)/len(psi_4d)**3, (batch[0]+1)*int(batch_num)/len(psi_4d)**3 # fix psi for activation
        big_psis = np.array(list(itertools.product(psi_4d[int(index_start):int(index_end)+1].tolist(), psi_4d.tolist(), psi_4d.tolist(), psi_4d.tolist()))).reshape(-1,16)
        run_psis = big_psis[int(np.ceil((index_start - int(index_start))*len(psi_4d)**3)): int((index_end - int(index_start))*len(psi_4d)**3)]

    elif "sparse-reg" in comment:
        index_start, index_end =int(batch[0]*batch_num), int((batch[0]+1)*batch_num)
        run_psis = psi_sparse[index_start:index_end]

    else:
        run_psis = default_reg

    params = [np.concatenate(q) for q in list(itertools.product( np.tile(inf_sample, (runs,1)).tolist(), run_psis.tolist()))]

    rng = np.random.default_rng(seed)
    seed_sequence = rng.bit_generator._seed_seq
    # Each set of parameters gets its own local random number generator.
    child_states = seed_sequence.spawn(len(params))
    child_rngs = (np.random.default_rng(state) for state in child_states)

    print(f'Running {len(params)} simulations in batch #{batch[0]}')

    batch_size = max(int(len(params) / (num_cpu + add_cpu)), 1)
    psi_list = Parallel(n_jobs=num_cpu + add_cpu, batch_size=batch_size)(delayed(lin_stoch_sim)(
        d_I = param[0], K_I = param[1], b_I = param[2], K_H = param[3], N_0 = param[4], I_0 = param[5],
        activation_regulation = param[-16:-12], NE_regulation = param[-12:-8], EM_regulation = param[-8:-4],
        contraction_regulation = param[-4:],
        infection_model = infection_model,
        reg_model = "mwc_like",
        duration = 1.5*sim_duration if 'long_sim' in comment else sim_duration,
        steps = 1.5*sim_steps if 'long_sim' in comment else sim_steps,
        seed=child_rng)
        for param, child_rng in zip(params, child_rngs)
    )

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
    parser.add_argument('--comment', dest='comment', type=str, required=False, default='sparse-reg',
                        help='simulation specifications')
    parser.add_argument('--infection_model', dest='infection_model', type=str, required=False, default='acute',
                        help='infection specifications')
    parser.add_argument('--seed', dest='seed', type=int, required=False, default=None,
                        help='The seed for the random number generator')
    args = parser.parse_args()

    os.chdir(args.outdir)

    run(args.batch, args.outdir, comment = args.comment, infection_model = args.infection_model, seed=args.seed)


if __name__ == '__main__':
    main()
