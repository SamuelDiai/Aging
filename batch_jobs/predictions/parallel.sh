#!/bin/bash
#targets=( "Age" )
#models=( "RandomForest" )
#datasets=( "UrineAndBlood" )

targets=( "Age" "Sex" )
models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" "NeuralNetwork" "ElasticNet" )
datasets=( "Anthropometry" "SpiroAndArterialAndBp" "ArterialAndBloodPressure" "ArterialStiffness" "AnthropometryBodySize" "BloodPressure" "Spirometry" )


outer_splits=10
inner_splits=5
n_iter=30

memory=8G
n_cores=4
time=30


for target in "${targets[@]}"
do
	for dataset in "${datasets[@]}"
	do
		for model in "${models[@]}"
		do
			if [ $target != "Sex" ] || [ $model != "ElasticNet" ]
			then
				for ((fold=0; fold <= $outer_splits-1; fold++))
				do
					job_name="${target}_${model}_${dataset}_${fold}.job"
					out_file="./logs/${target}_${model}_${dataset}_${fold}.out"
					err_file="./logs/${target}_${model}_${dataset}_${fold}.err"
					sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t 0-11:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold
				done
			else
				:
			fi
		done
	done
done
