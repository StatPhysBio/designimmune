import subprocess

import numpy as np

#from stoch_sim_model import *

# Create grid of regulation coeffs
m = 0
#regs = sample_grid(d=6,m=m)



for reg in regs:
    reg_str = ' '.join(reg.astype('U3'))
    command = f'sbatch submit_optimize_nets.sh {reg_str}'
    print(command)
    subprocess.Popen(command.split())
    break
