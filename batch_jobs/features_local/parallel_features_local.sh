#!/bin/bash
targets=( "Age" "Sex" )

models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" "ElasticNet ")
#models=( "ElasticNet" )
#datasets=( "Heart" "HeartPWA" "AnthropometryImpedance" "ECGAtRest" "AbdominalComposition" "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Brain" "HeartSize" "BodyComposition" "BoneComposition" )
datasets=( "Heart" "HeartPWA" "AnthropometryImpedance" "ECGAtRest" )
n_splits=2
n_iter=1


for target in "${targets[@]}"
do
	for model in "${models[@]}"
	do
		if [ $target != "Sex" ] || [ $model != "ElasticNet" ]
		then
			for dataset in "${datasets[@]}"
			do
			job_name="$target-$model-$dataset-features.job"

			out_file="./logs/$target-$model-$dataset-features.out"
			err_file="./logs/$target-$model-$dataset-features.err"
			sh batch_jobs/features_local/single_features_local.sh $model $n_iter $target $dataset $n_splits
			done
		else
			:
		fi
	done
done
