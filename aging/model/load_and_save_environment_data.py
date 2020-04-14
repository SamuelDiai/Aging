import pandas as pd
import glob
from ..environment_processing.base_processing import path_features , path_predictions, path_inputs_env, path_target_residuals
from ..environment_processing.disease_processing import read_infectious_diseases_data, read_infectious_diseases_antigens_data




target_dataset_to_field = {'AbdominalComposition' : 149,
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
                    'BloodPressure' : 100011,
                    'AnthropometryBodySize' : 100010,
                    'Anthropometry' : 100008,
                    'ArterialStiffness' : 100007,
                    'ArterialAndBloodPressure' : 'Custom',
                    'SpiroAndArterialAndBp' : 'Custom'
                    }

env_dataset_to_field = { 'InfectiousDiseaseAntigens' : 1307,
                         'InfectiousDiseases' : 51428

}


## Load ENV data

def load_data_env_(env_dataset, **kwargs):
    if env_dataset not in env_dataset_to_field.keys():
        raise ValueError('Wrong dataset name ! ')
    else :
        if dataset == 'InfectiousDiseaseAntigens':
            df = read_infectious_disease_antigens_data(**kwargs)
        elif dataset == 'InfectiousDiseases':
            df = read_infectious_diseases_data(**kwargs)
        return df

def load_data_env(env_dataset, **kwargs):
    selected_inputs = glob.glob(path_inputs_env + '%s.csv' % env_dataset)
    print(selected_inputs)
    if len(selected_inputs) == 0:
        print("Load New Data")
        df = load_data_env_(env_dataset, **kwargs)
        df.to_csv(path_inputs_env + dataset + '.csv')
        return df
    elif len(selected_inputs) == 1 :
        nrows = None
        if 'nrows' in kwargs.keys():
            nrows = kwargs['nrows']
        print("Load Existing Data")
        df = pd.read_csv(selected_inputs[0], nrows = nrows).set_index('eid')
        return df
    else :
        print("Error")
        raise ValueError('Too many Input file for the selected dataset')

## Load target data

def load_target_residuals(target_dataset, **kwargs):
    #target_dataset_without = target_dataset.replace('_', '')
    list_files = glob.glob(path_target_residuals + '%s.csv' % target_dataset)

    ## Select best model :
    if len(list_files) == 1 :
        df_organ = pd.read_csv(list_files[0]).set_index('eid')
    else :
        raise ValueError('')
    return df_organ[['residual', 'Sex']]


## Load FULL DATA

def load_data(env_dataset, target_dataset, **kwargs):

    df_env = load_data_env(env_dataset, **kwargs)
    df_target = load_target_residuals(target_dataset, **kwargs)
    df = df_env.join(df_target, how = 'inner')
    return df


## Saving
def save_features_to_csv(cols, features_imp, organ_target, dataset_env, model_name):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    final_df.set_index('features').to_csv(path_features + '/' + 'FeatureImp_' + dataset_env + '_' + organ_target + '_' + model_name + '.csv')


def save_predictions_to_csv(predicts_df, step, organ_target, dataset_env, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

    filename = 'Predictions_' + dataset_env + '_' + organ_target + '_' + str(target_dataset_to_field[dataset]) + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('eid').to_csv(path_predictions + '/' + filename)
