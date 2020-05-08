import pandas as pd
import glob
from string import ascii_uppercase

from ..processing.base_processing import read_ethnicity_data
from ..environment_processing.base_processing import path_features , path_predictions, path_inputs_env, path_target_residuals
from ..environment_processing.disease_processing import read_infectious_diseases_data, read_infectious_disease_antigens_data
from ..environment_processing.FamilyHistory import read_family_history_data
from ..environment_processing.HealthAndMedicalHistory import read_breathing_data, read_cancer_screening_data, read_chest_pain_data, read_claudication_data, read_eye_history_data, \
                                                             read_general_health_data, read_general_pain_data, read_hearing_data, read_medication_data, read_mouth_teeth_data
from ..environment_processing.LifestyleAndEnvironment import read_alcohol_data, read_diet_data, read_electronic_devices_data, read_physical_activity_data, read_sexual_factors_data,\
                                                             read_sleep_data, read_sun_exposure_data
from ..environment_processing.PsychosocialFactors import read_mental_health_data, read_social_support_data
from ..environment_processing.SocioDemographics import read_education_data, read_employment_data, read_household_data, read_other_sociodemographics_data
from ..environment_processing.HealthRelatedOutcomes import read_medical_diagnoses_data

dict_target_to_instance = {"Brain" : 2,
                           "UrineAndBlood" : 0,
                           "HeartPWA" : 2,
                           "Heart" : 2,
                           "Eye" :0,
                           "EyeIntraoculaPressure" :0,
                           "AnthropometryImpedance" :0,
                           "BrainGreyMatterVolumes" : 2,
                           "AnthropometryBodySize" : 0,
                           "UrineBiochemestry" :0,
                           "ArterialAndBloodPressure" :0,
                           "Spirometry" : 0,
                           "ECGAtRest" :2,
                           "EyeAutorefraction" :0,
                           "ArterialStiffness" : 0,
                           "BloodCount" : 0,
                           "BrainSubcorticalVolumes" :2,
                           "EyeAcuity" : 0,
                           "HeartSize" :2,
                           "BloodPressure" :0,
                           "SpiroAndArterialAndBp" :0,
                           "HeartImages" :2,
                           "BloodBiochemestry" : 0,
                           "Blood" : 0,
                           "Anthropometry" : 0,
                           "LiverImages" : 2}

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
                    'SpiroAndArterialAndBp' : 'Custom',
                    'LiverImages' : -1,
                    'HeartImages' : -2
                    }

env_dataset_to_field = { #'InfectiousDiseaseAntigens' : 1307,
                         #'InfectiousDiseases' : 51428,
                         'Alcohol' : 100051,
                         'Diet' : 100052,
                         'Education' : 100063,
                         'ElectronicDevices' : 100053,
                         'Employment' : 100064,
                         'FamilyHistory' : 100034,
                         'Eyesight' : 100041,
                         'Mouth' : 100046,
                         'GeneralHealth' : 100042,
                         'Breathing' : 100037,
                         'Claudification' : 100038,
                         'GeneralPain' : 100048,
                         'ChestPain' :100039,
                         'CancerScreening' : 100040,
                         'Medication': 100045,
                         'Hearing' : 100043,
                         'Household' : 100066,
                         'MentalHealth' : 100060,
                         'OtherSociodemographics' : 100067,
                         'PhysicalActivity' : 100054,
                         'SexualFactors' : 100056,
                         'Sleep' : 100057,
                         'SocialSupport' : 100061,
                         'SunExposure' : 100055,
}

medical_diagnoses_dict = dict(zip(['medical_diagnoses_%s' % letter for letter in ascii_uppercase], [41270 for letter in ascii_uppercase]))
env_dataset_to_field = {**env_dataset_to_field, **medical_diagnoses_dict}


## Load ENV data

def load_data_env_(env_dataset, **kwargs):
    if env_dataset not in env_dataset_to_field.keys():
        raise ValueError('Wrong dataset name ! ')
    else :
        if env_dataset == 'InfectiousDiseaseAntigens':
            df = read_infectious_disease_antigens_data(**kwargs)
        elif env_dataset == 'InfectiousDiseases':
            df = read_infectious_diseases_data(**kwargs)
        elif env_dataset == 'Alcohol':
            df = read_alcohol_data(**kwargs)
        elif env_dataset == 'Diet':
            df = read_diet_data(**kwargs)
        elif env_dataset == 'Education':
            df = read_education_data(**kwargs)
        elif env_dataset == 'ElectronicDevices':
            df = read_electronic_devices_data(**kwargs)
        elif env_dataset == 'Employment':
            df = read_employment_data(**kwargs)
        elif env_dataset == 'FamilyHistory':
            df = read_family_history_data(**kwargs)
        elif env_dataset == 'Eyesight':
            df = read_eye_history_data(**kwargs)
        elif env_dataset == 'Mouth':
            df = read_mouth_teeth_data(**kwargs)
        elif env_dataset == 'GeneralHealth':
            df = read_general_health_data(**kwargs)
        elif env_dataset == 'Breathing':
            df = read_breathing_data(**kwargs)
        elif env_dataset == 'Claudification':
            df = read_claudication_data(**kwargs)
        elif env_dataset == 'GeneralPain':
            df = read_general_pain_data(**kwargs)
        elif env_dataset == 'ChestPain':
            df = read_chest_pain_data(**kwargs)
        elif env_dataset == 'CancerScreening':
            df = read_cancer_screening_data(**kwargs)
        elif env_dataset == 'Medication':
            df = read_medication_data(**kwargs)
        elif env_dataset == 'Hearing':
            df = read_hearing_data(**kwargs)
        elif env_dataset == 'Household':
            df = read_household_data(**kwargs)
        elif env_dataset == 'MentalHealth':
            df = read_mental_health_data(**kwargs)
        elif env_dataset == 'OtherSociodemographics':
            df = read_other_sociodemographics_data(**kwargs)
        elif env_dataset == 'PhysicalActivity':
            df = read_physical_activity_data(**kwargs)
        elif env_dataset == 'SexualFactors':
            df = read_sexual_factors_data(**kwargs)
        elif env_dataset == 'Sleep':
            df = read_sleep_data(**kwargs)
        elif env_dataset == 'SocialSupport':
            df = read_social_support_data(**kwargs)
        elif env_dataset == 'SunExposure':
            df = read_sun_exposure_data(**kwargs)
        elif 'medical_diagnoses_' in env_dataset :
            letter = env_dataset.split('medical_diagnoses_')[1]
            df = read_medical_diagnoses_data(letter, **kwargs)
        return df


def load_ethnicity(**kwargs):
    """
    Load ethnicity data : must have eid as index
    """
    selected_inputs = glob.glob(path_inputs_env + 'ethnicities.csv')
    if len(selected_inputs) == 0:
        print("Load New Ethnicity Data")
        df_ethnicities = load_ethnicity_data(**kwargs)
        df_ethnicities.to_csv(path_inputs_env + 'ethnicities.csv')
        return df_ethnicities
    elif len(selected_inputs) == 1 :
        print("Load Existing Ethnicity Data")
        df_ethnicities = pd.read_csv(selected_inputs[0], **kwargs).set_index('eid')
        return df_ethnicities
    else :
        print("Error")
        raise ValueError('Too many input file for the Ethnicity dataset')


def load_data_env(env_dataset, **kwargs):
    selected_inputs = glob.glob(path_inputs_env + '%s.csv' % env_dataset)
    if len(selected_inputs) == 0:
        print("Load New Data")
        df = load_data_env_(env_dataset, **kwargs)
        df.to_csv(path_inputs_env + env_dataset + '.csv')
        return df
    elif len(selected_inputs) == 1 :
        print("Load Existing Data")
        df = pd.read_csv(selected_inputs[0], **kwargs).set_index('id')
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

        df_organ = pd.read_csv(list_files[0])
        if 'id' in df_organ.columns :
            df_organ = df_organ.set_index('id')
        elif 'id' not in  df_organ.columns and 'eid' in df_organ.columns :
            df_organ['id'] = df_organ['eid'].astype(str) + '_' + str(dict_target_to_instance[target_dataset])
            df_organ = df_organ.set_index('id')
    else :
        raise ValueError('')
    return df_organ[['residual', 'Sex', 'Age', 'eid']]


## Load FULL DATA

def load_data(env_dataset, target_dataset, **kwargs):
    """

    return dataframe with : 'residual', 'Age', 'Sex', 'eid' + env_features + ethnicty_features with id as index !

    """
    ## Join on id by default
    df_env = load_data_env(env_dataset, **kwargs)
    print(df_env.head())
    df_target = load_target_residuals(target_dataset, **kwargs)
    print(df_target.head())
    df_ethnicities = load_ethnicity(**kwargs)
    print(df_ethnicities.head())

    ## Try intersection
    df = df_env.join(df_target, how = 'inner', lsuffix='_dup', rsuffix='')
    columns_not_dup = df.columns[~df.columns.str.contains('_dup')]
    df = df[columns_not_dup]
    print(df.head())

    ## If empty intersection join on eid
    if df.shape[0] == 0:
        ## Change index :
        df_env = df_env.reset_index().set_index('eid')
        df_target = df_target.reset_index().set_index('eid')
        ## Join
        df = df_env.join(df_target, how = 'inner', lsuffix='_dup', rsuffix='_dup')
        ## Remove duplicates including id
        columns_not_dup = df.columns[~df.columns.str.contains('_dup')]
        df = df[columns_not_dup]
        ## Recreate id
        df['id'] = df.index
        df = df[columns_not_dup].reset_index().set_index('id')

    df = df.merge(df_ethnicities, on = 'eid', right_index = True)
    return df


## Saving
def save_features_to_csv(cols, features_imp, organ_target, dataset_env, model_name):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    final_df.set_index('features').to_csv(path_features + '/' + 'FeatureImp_' + dataset_env + '_' + organ_target + '_' + model_name + '.csv')


def save_predictions_to_csv(predicts_df, step, organ_target, dataset_env, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

    filename = 'Predictions_' + dataset_env + '_' + organ_target + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('eid').to_csv(path_predictions + '/' + filename)
