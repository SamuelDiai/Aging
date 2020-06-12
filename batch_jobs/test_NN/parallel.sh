#!/bin/bash
#datasets=( 'BloodBiochemestry' 'EyeAcuity' 'BraindMRIWeightedMeans' 'HearingTest' 'HearingTest' )
datasets=( 'HeartPWA' )
#architectures=( '1024;512;256;128' '512;256;128' '500;250' '300;100' '100;50' '30;10' )
architectures=( '30;10' )
for dataset in "${datasets[@]}"
do
  for architecture in "${architectures[@]}"
  do
    job_name="test_${dataset}_${architecture}.job"
    out_file="./logs/test_${dataset}_${architecture}.out"
    err_file="./logs/test_${dataset}_${architecture}.err"
    sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=16G -c 1 -p short -t 0-11:59 batch_jobs/test_NN/single.sh $dataset $architecture
  done
done
