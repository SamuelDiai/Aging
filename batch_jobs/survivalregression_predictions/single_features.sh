#!/bin/bash
#SBATCH -p medium
#SBATCH -t 4-23:59
#SBATCH --mail-user=samuel_diai@hms.harvard.edu


module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source /n/groups/patel/samuel/env_sam/bin/activate
python3 batch_jobs/survivalregression_predictions/python_caller_features.py $1 $2 $3 $4 $5
