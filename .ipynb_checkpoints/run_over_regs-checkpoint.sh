#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=285 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
rerun_list=(1 3 4 10 12 13 15 18 19 20 21 23 24 26 27 36 38 39 40 41 47 48 49 51 53 54 55 58 59 60 61 62 63 64 65 66 67 68 71 76 104 106 107 108 109 110 111 112 113 114 115 116 118 119 120 121 122 123 124 125 126 127 128 130 132 133 134 135 136 137 138 139 140 141 174 176 177 178 179 180 181 182 183 190 191 192 193 194 196 197 198 200 205 206 207 208 209 211 212 244 245 247 248 250 251 253 254 258 259 261 262 264 265 267 268 273 274 275 276 277 278 279 281 282)

# for i in "${rerun_list[@]}";

for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
