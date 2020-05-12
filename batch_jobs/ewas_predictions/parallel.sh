#!/bin/bash
models=( "Xgboost" )
#models=( "Xgboost" "RandomForest" "GradientBoosting" "LightGbm" "NeuralNetwork" "ElasticNet" )
target_datasets=( "LiverImages" )
#target_datasets=( "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Brain" "Heart" "HeartSize" "HeartPWA" "AnthropometryImpedance" "UrineBiochemestry" "BloodBiochemestry" "BloodCount" "Blood" "UrineAndBlood" "EyeAutorefraction" "EyeAcuity" "EyeIntraoculaPressure" "Eye" "Spirometry" "BloodPressure" "AnthropometryBodySize" "Anthropometry" "ArterialStiffness" "ArterialAndBloodPressure" "SpiroAndArterialAndBp" )
#input_datasets=( "Education" "medical_diagnoses_I" )
#input_datasets=( "Alcohol" "Diet" "Education" "ElectronicDevices" "Employment" "FamilyHistory" "Eyesight" "Mouth" "GeneralHealth" "Breathing" "Claudification" "GeneralPain" "ChestPain" "CancerScreening" "Medication" "Hearing" "Household" "MentalHealth" "OtherSociodemographics" "PhysicalActivity" "SexualFactors" "Sleep" "SocialSupport" "SunExposure" "medical_diagnoses_A" "medical_diagnoses_B" "medical_diagnoses_C" "medical_diagnoses_D" "medical_diagnoses_E" "medical_diagnoses_F" "medical_diagnoses_G" "medical_diagnoses_H" "medical_diagnoses_I" "medical_diagnoses_J" "medical_diagnoses_K" "medical_diagnoses_L" "medical_diagnoses_M" "medical_diagnoses_N" "medical_diagnoses_O" "medical_diagnoses_P" "medical_diagnoses_Q" "medical_diagnoses_R" "medical_diagnoses_S" "medical_diagnoses_T" "medical_diagnoses_U" "medical_diagnoses_V" "medical_diagnoses_W" "medical_diagnoses_X" "medical_diagnoses_Y" "medical_diagnoses_Z" )
input_datasets=( "Alcohol" )

outer_splits=3
inner_splits=2
n_iter=2
n_splits=2

memory=8G
n_cores=1



for target_dataset in "${target_datasets[@]}"
do
	for input_dataset in "${input_datasets[@]}"
	do
		job_name="${target_dataset}_${input_dataset}.job"
		out_file="./logs/${target_dataset}_${input_dataset}.out"
		err_file="./logs/${target_dataset}_${input_dataset}.err"

		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/linear_study.sh $target_dataset $input_dataset
		for model in "${models[@]}"
		do
			declare -a IDs=()
			for ((fold=0; fold <= $outer_splits-1; fold++))
			do
			  job_name="${target_dataset}_${model}_${input_dataset}_${fold}.job"
			  out_file="./logs/${target_dataset}_${model}_${input_dataset}_${fold}.out"
			  err_file="./logs/${target_dataset}_${model}_${input_dataset}_${fold}.err"
				ID=$(sbatch --parsable --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/ewas_predictions/single.sh $model $outer_splits $inner_splits $n_iter $target_dataset $input_dataset $fold)
				IDs+=($ID)
			done
			if [ $model != "NeuralNetwork" ]
			then
			   job_name="${target_dataset}_${model}_${input_dataset}_features.job"
				 out_file="./logs/${target_dataset}_${model}_${input_dataset}_features.out"
				 err_file="./logs/${target_dataset}_${model}_${input_dataset}_features.err"
         sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/ewas_predictions/single_features.sh $model $n_iter $target_dataset $input_dataset $n_splits
			else
					:
			fi
			job_name="${target_dataset}_${model}_${input_dataset}_postprocessing.job"
			out_file="./logs/${target_dataset}_${model}_${input_dataset}_postprocessing.out"
			err_file="./logs/${target_dataset}_${model}_${input_dataset}_postprocessing.err"

			printf -v joinedIDS '%s:' "${IDs[@]}"
			sbatch --dependency=afterok:${joinedIDS%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/postprocessing.sh $model $target_dataset $input_dataset $outer_splits
		done
	done
done
