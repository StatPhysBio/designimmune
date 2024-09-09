#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=752 # change depending on number of inputs: Nact-Ediv = 47, full-reg = 947
# if some jobs don't run, run the following:
rerun_list=(4 14 15 116 132 269 317 318)

for i in "${rerun_list[@]}";

# for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
