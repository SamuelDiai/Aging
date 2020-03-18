#!/bin/bash
targets=( "Age" )                                                                         
                                                               
#models=( "xgboost" "random_forest" "gradient_boosting" "lightgbm" )
models=( "elasticnet" )
datasets=( "abdominal_composition" "brain_grey_matter_volumes" "brain_subcortical_volumes" "brain" "heart_size" "body_composition" "bone_composition")
#datasets=( "heart_PWA" )
n_splits=5
n_iter=50


memory=8G                                                                                 
n_cores=4                                                                                 
time=30


for target in "${targets[@]}"                                                             
do                                                                                        
for model in "${models[@]}"                                                         
do
for dataset in "${datasets[@]}"  
do

job_name="$target-$model-$dataset-features.job"

out_file="./logs/$target-$model-$dataset-features.out"                                
err_file="./logs/$target-$model-$dataset-features.err"                                
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t 0-11:59 batch_jobs/single_features.sh $model $n_iter $target $dataset $n_splits
                                                                                     
done                                                                                      
done   
done
