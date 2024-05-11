import os

from joblib import Parallel, delayed
import numpy as np
import pickle

from stoch_sim_model import *

### Define function to run simulations and compute MI

# Set simulation parameters
runs = 10
vir_samp = np.tile(vir_prop, (runs,1)) # sample distribution of pathogen killing rate and size of naive repertoire
num_cpu = 25 # number of CPUs requested
batch_num = 80 # number of network variants to run on a cpu

def run(batch = 0, outdir='', virus_sample = vir_samp, infection_type = 'sec', comment = "act-reg-exp-reg"):

    for psi in psi_opts[int(batch[0])*batch_num:(int(batch[0])+1)*batch_num]:
    
        reg_coeff = psi
        outfile = ('-'.join((reg_coeff).astype('U4'))
                   + f'-{runs}-{infection_type}-'
                   + f'{comment}.npy')
        
        # Run simulations over different infections
        print(f'Running simulation with regs = {list(reg_coeff)}')
    
        runs_list = Parallel(n_jobs = num_cpu, batch_size = max(int(len(virus_sample)/num_cpu),1))(delayed(lin_stoch_sim)(d_I = param[0], 
                                                                                                                   K_IE = param[1],
                                                                                                                   K_EI = param[1]*param[2],
                                                                                                                   K_EH = K_IH*param[3],
                                                                                                                   memory_regulation = reg_coeff if "mem-reg" in comment else mem_psis,
                                                                                                                   activation_regulation = reg_coeff[0:3] if "act-reg" in comment else act_psis,
                                                                                                                   expansion_regulation = reg_coeff[3:] if "exp-reg" in comment else act_psis,
                                                                                                                   infection = infection_type,
                                                                                                                   vir_model = 'dep_harm')
                                                        for param in virus_sample)
    
        # Store data in dictionary
        runs_dict = {}
        select_keys = ['reg_coeffs','cell_time_series','time','prim_diff_bias', 'sec_diff_bias','lineage_diff', 'parameters', 'summary_stats']
        for k in select_keys:
            runs_dict[k] = list(d[k] for d in runs_list)

        # Compute and store mean and standard deviation
        mean_prim_diff_bias = []
        std_prim_diff_bias = []
        mean_sec_diff_bias = []
        std_sec_diff_bias = []
        mean_cell_series = []
        std_cell_series = []
        mean_sim_sum = []
        std_sim_sum = []
    
        parameters = np.array(runs_dict["parameters"])
        prim_diff_bias = np.array(runs_dict["prim_diff_bias"])
        sec_diff_bias = np.array(runs_dict["sec_diff_bias"])
        cell_series = np.array(runs_dict["cell_time_series"])
        sim_sum = np.array(runs_dict["summary_stats"])
    
        virs = np.unique(parameters[:,[4,7,13,14]], axis = 0)
        
        for i, vir in enumerate(virs): # Need to fix this to stop averaging over K_EI and K_EH
            index = (parameters[:,4] == vir[0])*(parameters[:,7] == vir[1])*(parameters[:,13] == vir[2])*(parameters[:,14] == vir[3])
    
            vir_parameters = np.unique(parameters[index], axis = 0).ravel()
            
            mean_sim_sum.append(np.concatenate((vir_parameters, np.mean(sim_sum[index], axis = 0))))
            std_sim_sum.append(np.concatenate((vir_parameters, np.std(sim_sum[index], axis = 0))))
    
            mean_prim_diff_bias.append(np.mean(prim_diff_bias[index], axis = 0))
            std_prim_diff_bias.append(np.std(prim_diff_bias[index], axis = 0))
    
            mean_sec_diff_bias.append(np.mean(sec_diff_bias[index], axis = 0))
            std_sec_diff_bias.append(np.std(sec_diff_bias[index], axis = 0))
            
            mean_cell_series.append(np.mean(cell_series[index], axis = 0))
            std_cell_series.append(np.std(cell_series[index], axis = 0))
    
        # Save datasets
        ### (1) Summary stats
        np.save(os.path.join(outdir, "summary_stats","mean",outfile), mean_sim_sum)
        np.save(os.path.join(outdir, "summary_stats","std",outfile), std_sim_sum)
        ### (2) Differentiation bias
        np.save(os.path.join(outdir, "prim_diff_bias","mean",outfile), mean_prim_diff_bias)
        np.save(os.path.join(outdir, "prim_diff_bias","std",outfile), std_prim_diff_bias)
        np.save(os.path.join(outdir, "sec_diff_bias","mean",outfile), mean_sec_diff_bias)
        np.save(os.path.join(outdir, "sec_diff_bias","std",outfile), std_sec_diff_bias)
        ### (3) Cell time series
        np.save(os.path.join(outdir, "cell_time_series","mean",outfile), mean_cell_series)
        np.save(os.path.join(outdir, "cell_time_series","std",outfile), std_cell_series)
    
        print(f'Saved {outfile}')

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--batch', dest='batch', nargs='+', type=float, required=True,
                        help='regulatory weights of the network')
    parser.add_argument('--outdir', dest='outdir', type=str, required=False, default='',
                        help='/PATH/TO/WHERE/OUTPUT/IS/SAVED')

    args = parser.parse_args()

    os.chdir(args.outdir)
    
    run(args.batch, args.outdir)


if __name__ == '__main__':
    main()