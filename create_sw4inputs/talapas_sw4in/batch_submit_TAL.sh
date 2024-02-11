#!/bin/bash
# loop over all submission files in the directory, 
# print the filename and submit the jobs to SLURM

# Syntax: bash batch_submit_TAL.sh

for FILE in *.srun; do
    echo ${FILE}
    sbatch ${FILE}

done

