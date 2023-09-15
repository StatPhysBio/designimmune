#!/bin/bash
DECIMAL_TO_3=({0..2}{0..2}{0..2}{0..2}{0..2}{0..2})
DECIMAL_TO_2=({0..1}{0..1}{0..1})

for i in {0..728};
do
    for j in {6..7}; # goes from 0 to 7
    do
        # Convert the number from decimal to base 3.
        REG="${DECIMAL_TO_3[i]}${DECIMAL_TO_2[j]}"
        
        # Add spaces between each digit.
        SPACED_REG=$(sed -e 's/./& /g' <(echo $REG))

        # Subtract 1 from each digit to get the correct reg number
        # and concatenate the results to a string.
        STR_REG=""
        for state in ${SPACED_REG[@]};
        do
            STR_REG="${STR_REG} $(bc -l <<< "$state")"
        done
        sbatch submit_optimize_nets.sh $i $STR_REG
    done
done
