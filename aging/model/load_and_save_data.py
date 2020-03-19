import pandas as pd
from ..processing .base_processing import path_features , path_predictions
from ..processing.abdominal_composition_processing import read_abdominal_data
from ..processing.brain_processing import read_grey_matter_volumes_data, read_subcortical_volumes_data, read_brain_data
from ..processing.heart_processing import read_heart_data, read_heart_size_data, read_heart_PWA_data
from ..processing.body_composition_processing import read_body_composition_data
from ..processing.bone_composition_processing import read_bone_composition_data 

dataset_to_field = {'AbdominalComposition' : 149, 
                    'BrainGreyMatterVolumes' : 1101,
                    'BrainSubcorticalVolumes': 1102,
                    'Brain' : 100,
                    'Heart' : 102,
                    'HeartSize' : 133,
                    'HeartPWA' : 128,
                    'BodyComposition' : 124,
                    'BoneComposition' : 125}

def load_data(dataset, **kwargs):
	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']
	if dataset not in dataset_to_field.keys():
		raise ValueError('Wrong dataset name ! ')
	else :
		if dataset == 'AbdominalComposition':
			df = read_abdominal_data(**kwargs)
		elif dataset == 'Brain':
			df = read_brain_data(**kwargs)
		elif dataset == 'BrainGreyMatterVolumes':
			df = read_grey_matter_volumes_data(**kwargs)
		elif dataset == 'BrainSubcorticalVolumes':
			df = read_subcortical_volumes_data(**kwargs)
		elif dataset == 'Heart':
			df = read_heart_data(**kwargs)
		elif dataset == 'HeartSize':
			df = read_heart_size_data(**kwargs) 
		elif dataset == 'HeartPWA':
			df = read_heart_PWA_data(**kwargs) 
		elif dataset == 'BoneComposition':
			df = read_bone_composition_data(**kwargs)
		elif dataset == 'BodyComposition':
			df = read_body_composition_data(**kwargs)
		return df  


def save_features_to_csv(cols, features_imp, target, dataset, model_name):
	final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
	final_df.set_index('features').to_csv(path_features + 'FeatureImp_' + target + '_' + dataset + '_' + model_name + '.csv')


def save_predictions_to_csv(predicts_df, step, target, dataset, model_name, fold, best_params):
	hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
	if len(best_params) != 7:
		hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

	filename = 'Predictions_' + target + '_' + dataset + '_' + str(dataset_to_field[dataset]) + '_main' +  '_raw' + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '_B.csv'
	predicts_df.to_csv(path_predictions + filename)

