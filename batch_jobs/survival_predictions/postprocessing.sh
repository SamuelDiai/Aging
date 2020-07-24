#!/bin/bash
#SBATCH -p short
#SBATCH -t 0-11:59
#SBATCH --mail-user=samuel_diai@hms.harvard.edu


module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source /n/groups/patel/samuel/env_sam/bin/activate
python3  batch_jobs/survival_predictions/python_caller_postprocessing.py $1 $2 $3 $4
