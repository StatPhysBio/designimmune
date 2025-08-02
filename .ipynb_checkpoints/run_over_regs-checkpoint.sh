#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=1248 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
rerun_list=(2 3 4 5 6 7 8 10 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 32 33 34 36 37 39 40 43 44 45 48 49 50 51 134 135 136 137 138 139 142 415 416 417 418 419 420 421 423 424 425 427 428 429 431 435 455 733 746 909 910 917)

for i in "${rerun_list[@]}";

# for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
