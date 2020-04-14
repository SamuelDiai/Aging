#!/bin/bash
#targets=( "Age" )
#models=( "RandomForest" )
#datasets=( "UrineAndBlood" )

target="Age"
model="Xgboost"
#datasets=( "Anthropometry" "SpiroAndArterialAndBp" "ArterialAndBloodPressure" "ArterialStiffness" "AnthropometryBodySize" "BloodPressure" "Spirometry" )
dataset="AnthropometryImpedance"

outer_splits=10
inner_splits=9
n_iter=30
fold=9
memory=8G
n_cores=1


job_name="${target}_${model}_${dataset}_${fold}.job"
out_file="./logs/${target}_${model}_${dataset}_${fold}.out"
err_file="./logs/${target}_${model}_${dataset}_${fold}.err"
sbatch  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold
