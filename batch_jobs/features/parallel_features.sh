#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" ) 
models=( "RandomForest" "GradientBoosting" "ElasticNet" )
#models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" "ElasticNet ")
#datasets=( "BraindMRIWeightedMeans" "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Brain" "Heart" "HeartSize" "HeartPWA" "BodyComposition" "BoneComposition" "ECGAtRest" "AnthropometryImpedance" "UrineBiochemestry" "BloodBiochemestry" "BloodCount" "Blood" "UrineAndBlood" "EyeAutorefraction" "EyeAcuity" "EyeIntraoculaPressure" "Eye" )
datasets=( "AnthropometryImpedance" )
n_splits=5
n_iter=50

memory=16G
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
			job_name="$target-$model-$dataset-features.job"

			out_file="./logs/$target-$model-$dataset-features.out"
			err_file="./logs/$target-$model-$dataset-features.err"
			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t 0-11:59 batch_jobs/features/single_features.sh $model $n_iter $target $dataset $n_splits
			done
		else
			:
		fi
	done
done
