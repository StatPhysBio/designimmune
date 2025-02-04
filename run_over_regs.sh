#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=2814 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
# rerun_list=(497)

# for i in "${rerun_list[@]}";

for ((i=2000; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
