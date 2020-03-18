#!/bin/bash
#targets=( "Sex" )
targets=( "Age" )                                                                         
#models=( "neural_network" )                                                                   
#models=( "xgboost" "random_forest" "gradient_boosting" "lightgbm" "neural_network")
models=( "elasticnet" )
datasets=( "abdominal_composition" "brain_grey_matter_volumes" "brain_subcortical_volumes" "heart_size" "heart_PWA" "body_composition" "bone_composition" "brain" )
#datasets=( "abdominal_composition" )
outer_splits=5
inner_splits=5
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
for ((fold=0; fold <= $outer_splits-1; fold++))
do

job_name="$target-$model-$dataset-$fold.job"

out_file="./logs/$target-$model-$dataset-$fold.out"                                
err_file="./logs/$target-$model-$dataset-$fold.err"                                
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t 0-11:59 batch_jobs/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold
done                                                                                      
done                                                                                      
done   

done
