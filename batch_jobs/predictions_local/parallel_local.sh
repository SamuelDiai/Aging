#!/bin/bash

targets=( "Age" "Sex" )                                                                         
                                                                  
#models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" "NeuralNetwork" "ElasticNet" )
models=( "Xgboost" "ElasticNet" )
datasets=( "AbdominalComposition" "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "HeartSize" "HeartPWA" "BodyComposition" "BoneComposition" "Brain" )
#datasets=( "BrainGreyMatterVolumes" "" )
outer_splits=2
inner_splits=2
n_iter=1



for target in "${targets[@]}"                                                             
do                                                                                        
	for model in "${models[@]}"                                                         
	do  
		if [[ $target != "Sex" || $model != "ElasticNet" ]]
		then 
			echo "OK ! "
			for dataset in "${datasets[@]}"  
			do
				for ((fold=0; fold <= $outer_splits-1; fold++))
				do                            
				sh batch_jobs/predictions_local/single_local.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold
				done                                                                                      
			done   	
		else 
			echo "NOT OK !"
		fi                                                                                   
	done   
done
