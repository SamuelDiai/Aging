#!/bin/bash
datasets=( 'BloodBiochemestry' 'EyeAcuity' 'BraindMRIWeightedMeans' 'HearingTest' 'HearingTest' )
architectures=( '1024;512;256;128' '512;256;128' '500;250' '300;100' '100;50' '30;10' )

for dataset in "${datasets[@]}"
do
  for architecture in "${architectures[@]}"
  do
    job_name="${architecture}_${dataset}.job"
    out_file="./logs/${architecture}_${dataset}.out"
    err_file="./logs/${architecture}_${dataset}.err"
    sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=16G -c 1 -p short -t 0-11:59 batch_jobs/test_NN/single.sh $dataset $architecture
  done
done
