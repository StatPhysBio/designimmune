#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=600 # change depending on number of inputs
# if some jobs don't run, run the following:
# rerun_list=(0 2 6 49 50 58 62 176 177 179 225 226 227 228 230 233 234 239 240 241 248 250 252 255 256 258 262 270 271 283 284 291 294 295 297 302 303 307 309 314 323 324 327 338 339 356 357 368 369 371 372 374 375 386 387 389 390 399 400 409 410 412 413 415 416 418 419 420 421 423 428 435 436 437 438 439 443 461 479 485 486 492 520 528 529)
#for i in "${rerun_list[@]}";

for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
