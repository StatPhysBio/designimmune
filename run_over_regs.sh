#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=284 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
rerun_list=(250 251 264 265 267 268 273 276 279 281 282)

for i in "${rerun_list[@]}";

# for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
