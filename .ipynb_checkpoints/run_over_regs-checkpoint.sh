#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

max_job=851 #2814 # change depending on number of inputs
# if some jobs don't run, run the following:
rerun_list=(11 55 531 622 625 664 734 741 742 743 744 749 750 751 752 753 758 759 760 761 776 784 785 786 793 794 795 801 802 803 804 819 826 827 828 829 835 836 837 843 844 845 846)

for i in "${rerun_list[@]}";

#for ((i=0; i<=${max_job}; i++));

do
    sbatch ${script_dir}/submit_app_job.sh "${script_dir}/submit_optimize_nets.sh $i"
done
