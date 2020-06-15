#!/bin/bash
#models=( "ElasticNet" )
models=( "LightGbm" "NeuralNetwork" "ElasticNet" )
#target_datasets=( "Alcohol" "Anthropometry" "AnthropometryBodySize" "AnthropometryImpedance" "ArterialAndBloodPressure" "ArterialStiffness" "Blood" "BloodBiochemestry" "BloodCount" "BloodPressure" "Brain" "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Breathing" "CancerScreening" "ChestPain" "Claudification" "Diet" "ECGAtRest" "Education" "ElectronicDevices" "Employment" "Eye" "EyeAcuity" "EyeAutorefraction" "EyeIntraoculaPressure" "Eyesight" "FamilyHistory" "GeneralHealth" "GeneralPain" "Hearing" "Heart" "HeartPWA" "HeartSize" "Household" "Medication" "MentalHealth" "Mouth" "OtherSociodemographics" "PhysicalActivity" "SexualFactors" "Sleep" "SocialSupport" "SpiroAndArterialAndBp" "Spirometry" "SunExposure" "UrineAndBlood" "UrineBiochemestry" )
target_datasets=( "HeartImages" "LiverImages" )
#target_datasets=( "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Brain" "Heart" "HeartSize" "HeartPWA" "AnthropometryImpedance" "UrineBiochemestry" "BloodBiochemestry" "BloodCount" "Blood" "UrineAndBlood" "EyeAutorefraction" "EyeAcuity" "EyeIntraoculaPressure" "Eye" "Spirometry" "BloodPressure" "AnthropometryBodySize" "Anthropometry" "ArterialStiffness" "ArterialAndBloodPressure" "SpiroAndArterialAndBp" )
#input_datasets=( "Education" )
#input_datasets=( "Alcohol" "Anthropometry" "AnthropometryBodySize" "AnthropometryImpedance" "ArterialAndBloodPressure" "ArterialStiffness" "Blood" "BloodBiochemestry" "BloodCount" "BloodPressure" "Brain" "BrainGreyMatterVolumes" "BrainSubcorticalVolumes" "Breathing" "CancerScreening" "ChestPain" "Claudification" "Diet" "ECGAtRest" "Education" "ElectronicDevices" "Employment" "Eye" "EyeAcuity" "EyeAutorefraction" "EyeIntraoculaPressure" "Eyesight" "FamilyHistory" "GeneralHealth" "GeneralPain" "Hearing" "Heart" "HeartPWA" "HeartSize" "Household" "Medication" "MentalHealth" "Mouth" "OtherSociodemographics" "PhysicalActivity" "SexualFactors" "Sleep" "SocialSupport" "SpiroAndArterialAndBp" "Spirometry" "SunExposure" "UrineAndBlood" "UrineBiochemestry" )

#input_datasets=( 'Alcohol' 'Diet' 'Education' 'ElectronicDevices' 'Employment' 'FamilyHistory' 'Eyesight' 'Mouth' 'GeneralHealth' 'Breathing' 'Claudification' 'GeneralPain' 'ChestPain' 'CancerScreening' 'Medication' 'Hearing' 'Household' 'MentalHealth' 'OtherSociodemographics' 'PhysicalActivity' 'SexualFactors' 'Sleep' 'SocialSupport' 'SunExposure' 'EarlyLifeFactors' )
input_datasets=( 'Alcohol' 'Diet' 'Education' 'ElectronicDevices' 'Employment' 'FamilyHistory' 'Eyesight' 'Mouth' 'GeneralHealth' 'Breathing' 'Claudification' 'GeneralPain' 'ChestPain' 'CancerScreening' 'Medication' 'Hearing' 'Household' 'MentalHealth' 'OtherSociodemographics' 'PhysicalActivity' 'SexualFactors' 'Sleep' 'SocialSupport' 'SunExposure' 'EarlyLifeFactors' )
search_dir_clusters='/n/groups/patel/samuel/EWAS/AutomaticClusters/'
search_dir_inputs='/n/groups/patel/samuel/EWAS/inputs_final'
outer_splits=5
inner_splits=4
n_iter=10
n_splits=5

memory=8G
n_cores=1

# declare -a IDsLoads=()
# for input_dataset in "${input_datasets[@]}"
# 	do
# 		job_name="Load_${input_dataset}.job"
# 		out_file="./logs/Load_${input_dataset}.out"
# 		err_file="./logs/Load_${input_dataset}.err"
# 		IDLoad=$(sbatch --parsable --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/load_datasets.sh $input_dataset)
# 		IDsLoads+=($IDLoad)
# 	done
#
# printf -v joinedIDsLoads '%s:' "${IDsLoads[@]}"
# job_name="Create_raw_data.job"
# out_file="./logs/Create_raw_data.out"
# err_file="./logs/Create_raw_data.err"
# ID_raw=$(sbatch --parsable --dependency=afterok:${joinedIDsLoads%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=50G -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/create_raw_data.sh)
#
#
# n_cores_inputing=50
# job_name="Input_data.job"
# out_file="./logs/Input_data.out"
# err_file="./logs/Input_data.err"
# ID_inputed=$(sbatch --parsable --dependency=afterok:$ID_raw --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=8G -c $n_cores_inputing -p short -t 0-11:59 batch_jobs/ewas_predictions/input_data.sh $n_cores_inputing)
# #ID_inputed=$(sbatch --parsable  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=8G -c 16 -p short -t 0-11:59 batch_jobs/ewas_predictions/input_data.sh $n_cores_inputing)
#
# ## To del :
# # n_cores_inputing=16
# # job_name="Input_data.job"
# # out_file="./logs/Input_data.out"
# # err_file="./logs/Input_data.err"
# # ID_inputed=$(sbatch --parsable --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=8G -c $n_cores_inputing -p short -t 0-11:59 batch_jobs/ewas_predictions/input_data.sh $n_cores_inputing)
#
# for target_dataset in "${target_datasets[@]}"
# do
#   ## Create Clusters
#   job_name="Create_clusters_${target_dataset}.job"
#   out_file="./logs/Create_clusters_${target_dataset}.out"
#   err_file="./logs/Create_clusters_${target_dataset}.err"
#   ID_cluster=$(sbatch --parsable --dependency=afterok:$ID_inputed --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=32 -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/create_clusters.sh $target_dataset)
#
#   search_dir_clusters_target=${search_dir_clusters}${target_dataset}
#
#   ## Linear EWAS
#   for input_dataset in "${search_dir_inputs}"/*
#   do
#     input_dataset_clean=$(basename ${input_dataset} .csv)
#     job_name="${target_dataset}_${input_dataset_clean}.job"
#     out_file="./logs/${target_dataset}_${input_dataset_clean}.out"
#     err_file="./logs/${target_dataset}_${input_dataset_clean}.err"
#     sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/linear_study.sh $target_dataset $input_dataset
#   done
#
#   ## Multivariate
#   for input_dataset in "$search_dir_clusters_target"/*
#   do
#     input_dataset_clean=$(basename ${input_dataset} .csv)
#     for model in "${models[@]}"
#     do
#       declare -a IDs=()
#       for ((fold=0; fold <= $outer_splits-1; fold++))
#       do
#         job_name="${target_dataset}_${model}_${input_dataset}_${fold}.job"
#         out_file="./logs/${target_dataset}_${model}_${input_dataset}_${fold}.out"
#         err_file="./logs/${target_dataset}_${model}_${input_dataset}_${fold}.err"
#       	ID=$(sbatch --parsable --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/ewas_predictions/single.sh $model $outer_splits $inner_splits $n_iter $target_dataset $input_dataset $fold)
#       	IDs+=($ID)
#       done
#       if [ $model != "NeuralNetwork" ]
#       then
#          job_name="${target_dataset}_${model}_${input_dataset}_features.job"
#       	 out_file="./logs/${target_dataset}_${model}_${input_dataset}_features.out"
#       	 err_file="./logs/${target_dataset}_${model}_${input_dataset}_features.err"
#         sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/ewas_predictions/single_features.sh $model $n_iter $target_dataset $input_dataset $n_splits
#       else
#       		:
#       fi
#       job_name="${target_dataset}_${model}_${input_dataset}_postprocessing.job"
#       out_file="./logs/${target_dataset}_${model}_${input_dataset}_postprocessing.out"
#       err_file="./logs/${target_dataset}_${model}_${input_dataset}_postprocessing.err"
#
#       printf -v joinedIDS '%s:' "${IDs[@]}"
#       sbatch --dependency=afterok:${joinedIDS%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/postprocessing.sh $model $target_dataset $input_dataset $outer_splits
#
#
#   done
#   ##
# done


n_cores_inputing=8
job_name="Input_data.job"
out_file="./logs/Input_data.out"
err_file="./logs/Input_data.err"
ID_inputed=$(sbatch --parsable  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=15G -c $n_cores_inputing -p medium -t 4-23:59 batch_jobs/ewas_predictions/input_data.sh $n_cores_inputing)
#ID_inputed=$(sbatch --parsable  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=8G -c 16 -p short -t 0-11:59 batch_jobs/ewas_predictions/input_data.sh $n_cores_inputing)


# for input_dataset in "${input_datasets[@]}"
# 	do
# 	for target_dataset in "${target_datasets[@]}"
# 	do
# 			job_name="${target_dataset}_${input_dataset}.job"
# 			out_file="./logs/${target_dataset}_${input_dataset}.out"
# 			err_file="./logs/${target_dataset}_${input_dataset}.err"
#
# 			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/linear_study.sh $target_dataset $input_dataset
#
# 		for model in "${models[@]}"
# 		do
# 			declare -a IDs=()
# 			for ((fold=0; fold <= $outer_splits-1; fold++))
# 			do
# 			  job_name="${target_dataset}_${model}_${input_dataset}_${fold}.job"
# 			  out_file="./logs/${target_dataset}_${model}_${input_dataset}_${fold}.out"
# 			  err_file="./logs/${target_dataset}_${model}_${input_dataset}_${fold}.err"
# 				ID=$(sbatch --parsable --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/ewas_predictions/single.sh $model $outer_splits $inner_splits $n_iter $target_dataset $input_dataset $fold)
# 				IDs+=($ID)
# 			done
# 			if [ $model != "NeuralNetwork" ]
# 			then
# 			   job_name="${target_dataset}_${model}_${input_dataset}_features.job"
# 				 out_file="./logs/${target_dataset}_${model}_${input_dataset}_features.out"
# 				 err_file="./logs/${target_dataset}_${model}_${input_dataset}_features.err"
#         sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/ewas_predictions/single_features.sh $model $n_iter $target_dataset $input_dataset $n_splits
# 			else
# 					:
# 			fi
# 			job_name="${target_dataset}_${model}_${input_dataset}_postprocessing.job"
# 			out_file="./logs/${target_dataset}_${model}_${input_dataset}_postprocessing.out"
# 			err_file="./logs/${target_dataset}_${model}_${input_dataset}_postprocessing.err"
#
# 			printf -v joinedIDS '%s:' "${IDs[@]}"
# 			sbatch --dependency=afterok:${joinedIDS%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ewas_predictions/postprocessing.sh $model $target_dataset $input_dataset $outer_splits
# 	done
# done
