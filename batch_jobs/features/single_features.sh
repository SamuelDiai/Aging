#!/bin/bash
#SBATCH -p short
#SBATCH -t 0-12:00
#SBATCH --mail-user=sasha_collin@hms.harvard.edu


module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source /home/al311/python_3.6.0/bin/activate
python3  batch_jobs/python_caller_features.py $1 $2 $3 $4 $5
