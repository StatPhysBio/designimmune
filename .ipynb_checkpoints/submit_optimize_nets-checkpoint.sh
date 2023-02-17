#!/bin/bash

#SBATCH --job-name=optimmune
#SBATCH -p ckpt
#SBATCH -A amath
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:59:00

d="/mmfs1/home/oukogu/github/infoimmune/opt_nets/"

if [ ! -d "$d" ]; then
  mkdir $d
fi

COMMAND="apptainer run --bind /gscratch /gscratch/spe/$USER/apptainer_images/maximmune.sif python"
$COMMAND optimize-nets.py --reg_coefs ${@:1} --outdir $d
