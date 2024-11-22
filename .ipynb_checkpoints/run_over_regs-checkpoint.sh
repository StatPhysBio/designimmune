#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=108 # change depending on number of inputs: Nact-Ediv = 47, full-reg = 947
# if some jobs don't run, run the following:
rerun_list=(289 340 344 348 363 364 367 379 383 387)

# for i in "${rerun_list[@]}";

for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
