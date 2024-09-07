#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=752 # change depending on number of inputs: Nact-Ediv = 47, full-reg = 947
# if some jobs don't run, run the following:
# rerun_list=(74 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 95 98 102 109 110 111 112 113 114 118 119 120 121 122 123 124 125 126 127 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 238 239 240 241 242 243 293 301 315 147)

# for i in "${rerun_list[@]}";

for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
