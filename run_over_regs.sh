#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=851 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
rerun_list=(207 208 465 475 486 487 488 489 490 491 493 500 502 503 504 505 506 507 508 509 511 512 514 515 516 518 525 544 692 711 768)

for i in "${rerun_list[@]}";

#for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
