#!/bin/bash

# Done :
target_datasets=( '*' '*instances01' '*instances1.5x' '*instances23' 'Abdomen' 'AbdomenLiver' 'AbdomenPancreas' 'Arterial' 'ArterialPulseWaveAnalysis' 'ArterialCarotids' 'Biochemistry' 'BiochemistryUrine' 'BiochemistryBlood' 'Brain' 'BrainCognitive' 'BrainMRI' 'Eyes' 'EyesAll' 'EyesFundus' 'EyesOCT' 'Hearing' 'Heart' 'HeartECG' 'HeartMRI' 'ImmuneSystem' 'Lungs' 'Musculoskeletal' 'MusculoskeletalSpine' 'MusculoskeletalHips' 'MusculoskeletalKnees' 'MusculoskeletalFullBody' 'MusculoskeletalScalars' 'PhysicalActivity' )
input_datasets=( 'Alcohol' 'Diet' 'Education' 'ElectronicDevices' 'Employment' 'FamilyHistory' 'Eyesight' 'Mouth' 'GeneralHealth' 'Breathing' 'Claudification' 'GeneralPain' 'ChestPain' 'CancerScreening' 'Medication' 'Hearing' 'Household' 'MentalHealth' 'OtherSociodemographics' 'PhysicalActivityQuestionnaire' 'SexualFactors' 'Sleep' 'SocialSupport' 'SunExposure' 'EarlyLifeFactors' 'Smoking' 'AnthropometryBodySize' 'AnthropometryImpedance' 'ArterialStiffness' 'Biochemistry' 'BloodBiochemistry' 'BloodCount' 'BloodPressure' 'BoneDensitometryOfHeel' 'BrainAndCognitive' 'BraindMRIWeightedMeans' 'BrainGreyMatterVolumes' 'BrainMRIAllBiomarkers' 'BrainSubcorticalVolumes' 'CarotidUltrasound' 'CognitiveAllBiomarkers' 'CognitiveFluidIntelligence' 'CognitiveMatrixPatternCompletion' 'CognitiveNumericMemory' 'CognitivePairedAssociativeLearning' 'CognitivePairsMatching' 'CognitiveProspectiveMemory' 'CognitiveReactionTime' 'CognitiveSymbolDigitSubstitution' 'CognitiveTowerRearranging' 'CognitiveTrailMaking' 'ECGAtRest' 'EyeAcuity' 'EyeAutorefraction' 'EyeIntraocularPressure' 'VascularAllBiomarkers' 'UrineBiochemistry' 'Spirometry' 'PhysicalActivity' 'MusculoskeletalAllBiomarkers' 'HeartSize' 'HeartPWA' 'EyesAllBiomarkers' 'HandGripStrength' 'HearingTest' 'HeartAllBiomarkers' 'HeartMRIAll' 'ENSEMBLE_HealthAndMedicalHistory' 'ENSEMBLE_LifestyleAndEnvironment' 'ENSEMBLE_PsychosocialFactors' 'ENSEMBLE_SocioDemographics' )


memory=16G
n_cores=1


for input_dataset in "${input_datasets[@]}"
	do
	for target_dataset in "${target_datasets[@]}"
	do
			job_name="create_cluster_${target_dataset}_${input_dataset}.job"
			out_file="./logs/create_cluster_${target_dataset}_${input_dataset}.out"
			err_file="./logs/create_cluster_${target_dataset}_${input_dataset}.err"

			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ClusterEWAS/single.sh $target_dataset $input_dataset
	done
done
