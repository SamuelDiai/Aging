#!/bin/bash
targets=( "Age" "Sex" )                                                                         
                                                               
#models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" )
models=( "ElasticNet" )
datasets=( "AbdominalComposition" "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Brain" "HeartSize" "BodyComposition" "BoneComposition")
#datasets=( "HeartPWA" )
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
