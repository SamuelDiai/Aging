#!/bin/bash

targets=( "Sex" "Age" )
models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" "NeuralNetwork" "ElasticNet" )
#models=( "Xgboost" )
#datasets=( "Heart" "HeartPWA" "AnthropometryImpedance" "ECGAtRest" "AbdominalComposition" "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Brain" "HeartSize" "BodyComposition" "BoneComposition" )
datasets=( "Heart" "HeartPWA" "AnthropometryImpedance" "ECGAtRest" )


outer_splits=5
inner_splits=5
n_iter=30

memory=8G
n_cores=4
time=30


for target in "${targets[@]}"
do
	for model in "${models[@]}"
	do
		if [ $target != "Sex" ] || [ $model != "ElasticNet" ]
		then
			for dataset in "${datasets[@]}"
			do
				for ((fold=0; fold <= $outer_splits-1; fold++))
				do
					job_name="$target-$model-$dataset-$fold.job"
					out_file="./logs/$target-$model-$dataset-$fold.out"
					err_file="./logs/$target-$model-$dataset-$fold.err"
					sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t 0-11:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold
				done
			done
		else
			:
		fi
	done

done
