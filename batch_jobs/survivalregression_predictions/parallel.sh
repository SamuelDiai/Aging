#!/bin/bash

models=( "LightGbm" )
#models=( "LightGbm" "ElasticNet" "NeuralNetwork" )
#datasets=( 'HandGripStrength' 'BrainGreyMatterVolumes' 'BrainSubcorticalVolumes' 'HeartSize' 'HeartPWA' 'ECGAtRest' 'AnthropometryImpedance' 'UrineBiochemestry' 'BloodBiochemestry' 'BloodCount' 'EyeAutorefraction' 'EyeAcuity' 'EyeIntraoculaPressure' 'BraindMRIWeightedMeans' 'Spirometry' 'BloodPressure' 'AnthropometryBodySize' 'ArterialStiffness' 'CarotidUltrasound' 'BoneDensitometryOfHeel' 'HearingTest' )
#datasets=( 'HandGripStrength' 'BrainSubcorticalVolumes' 'HeartSize' 'HeartPWA' 'ECGAtRest' 'AnthropometryImpedance' 'UrineBiochemestry' )
datasets=( "CognitiveAllBiomarkers" )
targets=( "All" "CVD" "Cancer" )
outer_splits=10
inner_splits=9
n_iter=30
n_splits=10

memory=8G
n_cores=1


for target in "${targets[@]}"
do
	for model in "${models[@]}"
	do
		for dataset in "${datasets[@]}"
		do
			declare -a IDs=()
			for ((fold=0; fold <= $outer_splits-1; fold++))
			do
				job_name="${target}_${model}_${dataset}_${fold}.job"
				out_file="./logs/${target}_${model}_${dataset}_${fold}.out"
				err_file="./logs/${target}_${model}_${dataset}_${fold}.err"

				# To del :
				ID=$(sbatch --parsable  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/survivalregression_predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold)
				IDs+=($ID)
			done

			# job_name="${target}_${model}_${dataset}_features.job"
			# out_file="./logs/${target}_${model}_${dataset}_features.out"
			# err_file="./logs/${target}_${model}_${dataset}_features.err"
			#
			# # To del :
			#
			# sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/survivalregression_predictions/single_features.sh $model $n_iter $target $dataset $n_splits
			#
			# job_name="${target}_${model}_${dataset}_postprocessing.job"
			# out_file="./logs/${target}_${model}_${dataset}_postprocessing.out"
			# err_file="./logs/${target}_${model}_${dataset}_postprocessing.err"
			#
			# printf -v joinedIDS '%s:' "${IDs[@]}"
			# sbatch --dependency=afterok:${joinedIDS%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/survivalregression_predictions/postprocessing.sh $model $target $dataset $outer_splits
			#

		done
	done
done
