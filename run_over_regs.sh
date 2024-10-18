#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=649 # change depending on number of inputs: Nact-Ediv = 47, full-reg = 947
# if some jobs don't run, run the following:
rerun_list=(536 537 539 540 541 545 546 550 552 553 559 561 562 563 566 567 568 571 572 574 575 576 578 579 580 581 583 584 585 592 593 596 597 599 600 602 604 605 606 607 608 610 612 613 614 615 616 626 627 630 632 635 637 639 640 641 642 644 645 646 647 648)

for i in "${rerun_list[@]}";

# for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
