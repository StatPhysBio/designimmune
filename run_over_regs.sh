#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=854 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
rerun_list=(0 1 2 19 20 22 23 24 25 26 29 30 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 132 133 135 136 137 138 139 140 146 147 148 149 150 151 152 153 156 157 158 159 160 161 162 163 164 165 166 167 168)

# for i in "${rerun_list[@]}";

for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
