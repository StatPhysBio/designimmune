#!/bin/bash
DECIMAL_TO_3=({0..2}{0..2}{0..2}{0..2}{0..2}{0..2})

for i in {0..728};
do
    # Convert the number from decimal to base 3.
    REG=${DECIMAL_TO_3[i]}

    # Add spaces between each digit.
    SPACED_REG=$(sed -e 's/./& /g' <(echo $REG))

    # Divide each digit by 2 to get the correct reg number
    # and concatenate the results to a string.
    STR_REG=""
    for state in ${SPACED_REG[@]};
    do
        STR_REG="${STR_REG} $(bc -l <<< "$state / 2")"
    done

   sbatch submit_optimize_nets.sh $i $STR_REG
done
