#!/bin/bash
#targets=( "Age" )
#models=( "RandomForest" )
#datasets=( "UrineAndBlood" )

#targets=( "Age" "Sex" )
#models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" "NeuralNetwork" "ElasticNet" )
#datasets=( "Anthropometry" "SpiroAndArterialAndBp" "ArterialAndBloodPressure" "ArterialStiffness" "AnthropometryBodySize" "BloodPressure" "Spirometry" )
datasets=( 'HandGripStrength' 'BrainGreyMatterVolumes' 'BrainSubcorticalVolumes' 'HeartSize' 'HeartPWA' 'ECGAtRest' 'AnthropometryImpedance' 'UrineBiochemestry' 'BloodBiochemestry' 'BloodCount' 'EyeAutorefraction' 'EyeAcuity' 'EyeIntraoculaPressure' 'BraindMRIWeightedMeans' 'Spirometry' 'BloodPressure' 'AnthropometryBodySize' 'ArterialStiffness' 'CarotidUltrasound' 'BoneDensitometryOfHeel' 'HearingTest' )

outer_splits=10
inner_splits=9
n_iter=30
n_splits=5

memory=8G
n_cores=1



fold=0
model='ElasticNet'
target='Age'
for dataset in "${datasets[@]}"
do
	job_name="${target}_${model}_${dataset}_${fold}.job"
  out_file="./logs/${target}_${model}_${dataset}_${fold}.out"
	err_file="./logs/${target}_${model}_${dataset}_${fold}.err"
	sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold
done



# for target in "${targets[@]}"
# do
# 	for dataset in "${datasets[@]}"
# 	do
# 		for model in "${models[@]}"
# 		do
# 			if [ $target != "Sex" ] || [ $model != "ElasticNet" ]
# 			then
# 				declare -a IDs=()
# 				for ((fold=0; fold <= $outer_splits-1; fold++))
# 				do
# 					job_name="${target}_${model}_${dataset}_${fold}.job"
# 					out_file="./logs/${target}_${model}_${dataset}_${fold}.out"
# 					err_file="./logs/${target}_${model}_${dataset}_${fold}.err"
# 					ID=$(sbatch --parsable --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold)
# 					IDs+=($ID)
# 				done
#
# 				if [ $model != "NeuralNetwork" ]
# 				then
# 					job_name="${target}_${model}_${dataset}_features.job"
# 					out_file="./logs/${target}_${model}_${dataset}_features.out"
# 					err_file="./logs/${target}_${model}_${dataset}_features.err"
#
# 					sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single_features.sh $model $n_iter $target $dataset $n_splits
# 				else
# 					:
# 				fi
#
# 				job_name="${target}_${model}_${dataset}_postprocessing.job"
# 				out_file="./logs/${target}_${model}_${dataset}_postprocessing.out"
# 				err_file="./logs/${target}_${model}_${dataset}_postprocessing.err"
#
# 				printf -v joinedIDS '%s:' "${IDs[@]}"
# 				sbatch --dependency=afterok:${joinedIDS%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/predictions/postprocessing.sh $model $target $dataset $outer_splits
# 			else
# 				:
# 			fi
#
# 		done
# 	done
# done
