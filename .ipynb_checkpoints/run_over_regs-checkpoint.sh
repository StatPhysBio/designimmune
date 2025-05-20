#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=295 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
# rerun_list=(1 2 29 34 52 53 59 61 66 74 77 86 87 88 89 90 95 98 101 102 103 106 107 116 117 118 119 124 131 136 137 138 141 145 292 308 311)

# for i in "${rerun_list[@]}";

for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
