#!/bin/bash

#SBATCH --job-name=optimmune
#SBATCH -p ckpt 
#SBATCH -A amath 
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=20
#SBATCH --time=3:59:00
#SBATCH --output=/gscratch/scrubbed/oukogu/slurm_output/%j.out

SLURM_OUTDIR=/gscratch/scrubbed/oukogu/slurm_output/

d="/mmfs1/home/oukogu/github/infoimmune/opt_nets/"

if [ ! -d "$d" ]; then
  mkdir $d
fi

COMMAND="apptainer run --bind /gscratch /gscratch/spe/$USER/apptainer_images/maximmune.sif python"
$COMMAND optimize-nets.py --reg_coefs ${@:2} --outdir $d

mv ${SLURM_OUTDIR}${SLURM_JOB_ID}.out ${SLURM_OUTDIR}net-${1}.out
