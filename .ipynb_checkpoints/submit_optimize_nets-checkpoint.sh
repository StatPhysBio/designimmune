#!/bin/bash

#SBATCH --job-name=optimmune
#SBATCH -p ckpt 
#SBATCH -A amath 
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
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

arg=${1}


#COMMAND="apptainer run --bind /gscratch /gscratch/spe/$USER/apptainer_images/maximmune.sif python"
#$COMMAND optimize_nets.py --reg_coefs $reg_coeffs --reg_logs $reg_logs --outdir $d
python optimize_nets.py --reg_opt $arg --outdir $d

mv ${SLURM_OUTDIR}${SLURM_JOB_ID}.out ${SLURM_OUTDIR}net-${1}.out
