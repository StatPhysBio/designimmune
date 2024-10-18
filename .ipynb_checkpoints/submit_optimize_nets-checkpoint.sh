#!/bin/bash
#SBATCH --job-name=optimmune
#SBATCH -p ckpt 
#SBATCH -A amath 
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --ntasks-per-node=38
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/oukogu/slurm_output/%j.out

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${script_dir}/source_app_mamba.sh
mamba activate maximmune

SLURM_OUTDIR=/gscratch/scrubbed/oukogu/slurm_output/
d="/gscratch/scrubbed/oukogu/infoimmune/sim_output/no_cell_var/raw"
comment="sparse-reg"

if [ ! -d "$d" ]; then
  mkdir $d
fi

arg=${1}


#COMMAND="apptainer run --bind /gscratch /gscratch/spe/$USER/apptainer_images/maximmune.sif python"
#$COMMAND optimize_nets.py --batch $batch  --outdir $d --comment $comment
python optimize_nets.py --batch $arg --outdir $d --comment $comment

mv ${SLURM_OUTDIR}${SLURM_JOB_ID}.out ${SLURM_OUTDIR}batch-${1}.out
