#!/bin/bash
#SBATCH --job=optimmune
#SBATCH --account=spe
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=200GB
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/oukogu/slurm_output/%j.out

# The bash file and its arguments need to be passed in surrounded by quotes.
# E.g., sbatch submit_app_job.sh "/FULL/PATH/TO/BASH_FILE.sh ARG1 ARG2 ARG3"
# Additionally, the bash file to be executed should have the executable permission:
# chmod 744 /FULL/PATH/TO/BASH_FILE.sh

apptainer exec --bind /gscratch \
    --overlay /gscratch/spe/${USER}/apptainer_images/mamba-overlay.img:ro \
    /gscratch/spe/${USER}/apptainer_images/hyak-container.sif /bin/bash -l -c "${@:1}"
