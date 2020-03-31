import pandas as pd
from ..processing .base_processing import path_features , path_predictions, path_inputs
from ..processing.abdominal_composition_processing import read_abdominal_data
from ..processing.brain_processing import read_grey_matter_volumes_data, read_subcortical_volumes_data, read_brain_data, read_brain_dMRI_weighted_means_data
from ..processing.heart_processing import read_heart_data, read_heart_size_data, read_heart_PWA_data
from ..processing.body_composition_processing import read_body_composition_data
from ..processing.bone_composition_processing import read_bone_composition_data
from ..processing.ecg_processing import read_ecg_at_rest_data
from ..processing.anthropometry_processing import read_anthropometry_impedance_data
from ..processing.biochemestry_processing import read_blood_biomarkers_data, read_urine_biomarkers_data, read_blood_count_data, read_blood_data, read_urine_and_blood_data
from ..processing.eye_processing import read_eye_autorefraction_data, read_eye_acuity_data, read_eye_intraocular_pressure_data, read_eye_data
from ..processing.spirometry_processing import read_spirometry_data
from ..processing.blood_pressure_processing import read_blood_pressure_data

dataset_to_field = {'AbdominalComposition' : 149,
                    'BrainGreyMatterVolumes' : 1101,
                    'BrainSubcorticalVolumes': 1102,
                    'Brain' : 100,
                    'Heart' : 102,
                    'HeartSize' : 133,
                    'HeartPWA' : 128,
                    'BodyComposition' : 124,
                    'BoneComposition' : 125,
                    'ECGAtRest' : 12657,
                    'AnthropometryImpedance' : 100008,
                    'UrineBiochemestry' : 100083,
                    'BloodBiochemestry' : 17518,
                    'BloodCount' : 100081, # Need to do blood infection
                    'Blood' : 100080,
                    'UrineAndBlood' : 'Custom',
                    'EyeAutorefraction' : 100014,
                    'EyeAcuity' : 100017,
                    'EyeIntraoculaPressure' : 100015,
                    'Eye' : 100013,
                    'BraindMRIWeightedMeans' : 135,
                    'Spirometry' :  100020,
                    'BloodPressure' : 100011
                    }

def load_data_(dataset, **kwargs):
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
        elif dataset == 'ECGAtRest':
            df = read_ecg_at_rest_data(**kwargs)
        elif dataset == 'AnthropometryImpedance':
            df = read_anthropometry_impedance_data(**kwargs)
        elif dataset == 'UrineBiochemestry':
            df = read_urine_biomarkers_data(**kwargs)
        elif dataset == 'BloodBiochemestry':
            df = read_blood_biomarkers_data(**kwargs)
        elif dataset == 'BloodCount':
            df = read_blood_count_data(**kwargs)
        elif dataset == 'Blood':
            df = read_blood_data(**kwargs)
        elif dataset == 'UrineAndBlood':
            df = read_urine_and_blood_data(**kwargs)
        elif dataset == 'EyeAutorefraction':
            df = read_eye_autorefraction_data(**kwargs)
        elif dataset == 'EyeIntraoculaPressure':
            df = read_eye_intraocular_pressure_data(**kwargs)
        elif dataset == 'EyeAcuity':
            df = read_eye_acuity_data(**kwargs)
        elif dataset == 'Eye':
            df = read_eye_data(**kwargs)
        elif dataset == 'BraindMRIWeightedMeans':
            df = read_brain_dMRI_weighted_means_data(**kwargs)
        elif dataset == 'BloodPressure':
            df = read_blood_pressure_data(**kwargs)
        elif dataset == 'Spirometry':
            df = read_spirometry_data(**kwargs)
        return df


def load_data(dataset, **kwargs):
    list_inputs = glob.glob(path_inputs + '*.csv')
    selected_inputs = [elem for elem in list_inputs if dataset + '.csv' in elem]
    if len(selected_inputs) == 0:
        df = load_data_(dataset, **kwargs)
        df.to_csv(path_inputs + dataset + '.csv')
        return df
    elif len(selected_inputs) == 1 :
        df = pd.read_csv(selected_inputs[0]).set_index('eid')
        return df
    else :
        raise ValueError('Too many Input file for the selected dataset')

def save_features_to_csv(cols, features_imp, target, dataset, model_name):
	final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
	final_df.set_index('features').to_csv(path_features + '/' + 'FeatureImp_' + target + '_' + dataset + '_' + model_name + '.csv')


def save_predictions_to_csv(predicts_df, step, target, dataset, model_name, fold, best_params):
	hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
	if len(best_params) != 7:
		hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

	filename = 'Predictions_' + target + '_' + dataset + '_' + str(dataset_to_field[dataset]) + '_main' +  '_raw' + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '_B_withoutRF.csv'
	predicts_df.set_index('eid').to_csv(path_predictions + '/' + filename)
