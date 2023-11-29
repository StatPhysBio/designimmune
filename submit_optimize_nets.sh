#!/bin/bash

#SBATCH --job-name=optimmune
#SBATCH -p ckpt 
#SBATCH -A amath 
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:59:00
#SBATCH --output=/gscratch/scrubbed/oukogu/slurm_output/%j.out

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${script_dir}/source_app_mamba.sh
mamba activate maximmune

SLURM_OUTDIR=/gscratch/scrubbed/oukogu/slurm_output/
#d="/mmfs1/home/oukogu/github/infoimmune/opt_nets/"
d="/gscratch/scrubbed/oukogu/infoimmune/sim_output/no_cell_var/raw/"

if [ ! -d "$d" ]; then
  mkdir $d
fi

# Collect all the arguments after the first into an array
args=${@:2}

# The list contains numbers separated by spaces. To capture the first 6 digits,
# 12 characters must be taken.
reg_coeffs=${args:0:11}
# The rest of the digits separated by spaces are the reg log parameters.
reg_logs=${args:11}

#COMMAND="apptainer run --bind /gscratch /gscratch/spe/$USER/apptainer_images/maximmune.sif python"
#$COMMAND optimize-nets.py --reg_coefs $reg_coeffs --reg_logs $reg_logs --outdir $d
python optimize-nets.py --reg_coefs $reg_coeffs --reg_logs $reg_logs --outdir $d

mv ${SLURM_OUTDIR}${SLURM_JOB_ID}.out ${SLURM_OUTDIR}net-${1}-${reg_logs}.out
