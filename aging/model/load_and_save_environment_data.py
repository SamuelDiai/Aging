import pandas as pd
import glob
from string import ascii_uppercase
from functools import partial

from ..processing.base_processing import read_ethnicity_data
from ..environment_processing.base_processing import path_features , path_predictions, path_inputs_env, path_target_residuals, ETHNICITY_COLS
from ..environment_processing.disease_processing import read_infectious_diseases_data, read_infectious_disease_antigens_data
from ..environment_processing.FamilyHistory import read_family_history_data
from ..environment_processing.HealthAndMedicalHistory import read_breathing_data, read_cancer_screening_data, read_chest_pain_data, read_claudication_data, read_eye_history_data, \
                                                             read_general_health_data, read_general_pain_data, read_hearing_data, read_medication_data, read_mouth_teeth_data
from ..environment_processing.LifestyleAndEnvironment import read_alcohol_data, read_diet_data, read_electronic_devices_data, read_physical_activity_data, read_sexual_factors_data,\
                                                             read_sleep_data, read_sun_exposure_data
from ..environment_processing.PsychosocialFactors import read_mental_health_data, read_social_support_data
from ..environment_processing.SocioDemographics import read_education_data, read_employment_data, read_household_data, read_other_sociodemographics_data
from ..environment_processing.HealthRelatedOutcomes import read_medical_diagnoses_data

dict_target_to_instance_and_id = {"Brain" : (2, 100),
                           "UrineAndBlood" : (0, 'Custom'),
                           "HeartPWA" : (2, 128),
                           "Heart" : (2, 102),
                           "Eye" : (0, 100013),
                           "EyeIntraoculaPressure" : (0, 100015),
                           "AnthropometryImpedance" : (0, 100008),
                           "BrainGreyMatterVolumes" : (2, 1101),
                           "AnthropometryBodySize" : (0, 100010),
                           "UrineBiochemestry" : (0, 100083),
                           "ArterialAndBloodPressure" : (0, 'Custom'),
                           "Spirometry" : (0, 100020),
                           "ECGAtRest" : (2, 12657),
                           "EyeAutorefraction" : (0, 100014),
                           "ArterialStiffness" : (0, 100007),
                           "BloodCount" : (0, 100081),
                           "BrainSubcorticalVolumes" : (2, 1102),
                           "EyeAcuity" : (0, 100017),
                           "HeartSize" : (2, 133),
                           "BloodPressure" : (0, 100011),
                           "SpiroAndArterialAndBp" : (0, 'Custom'),
                           "HeartImages" : (2, 'Alan'),
                           "BloodBiochemestry" : (0, 17518),
                           "Blood" : (0, 100080),
                           "Anthropometry" : (0, 100008),
                           "LiverImages" : (2, 'Alan')
                           }



map_envdataset_to_dataloader_and_field = {
    'Alcohol' : (read_alcohol_data, 100051),
    'Diet' : (read_diet_data, 100052),
    'Education' : (read_education_data, 100063),
    'ElectronicDevices' : (read_electronic_devices_data, 100053),
    'Employment' : (read_employment_data, 100064),
    'FamilyHistory' : (read_family_history_data, 100034),
    'Eyesight' : (read_eye_history_data, 100041),
    'Mouth' : (read_mouth_teeth_data, 100046),
    'GeneralHealth' : (read_general_health_data, 100042),
    'Breathing' : (read_breathing_data, 100037),
    'Claudification' : (read_claudication_data, 100038),
    'GeneralPain' : (read_general_pain_data, 100048),
    'ChestPain' : (read_chest_pain_data, 100039),
    'CancerScreening' : (read_cancer_screening_data, 100040),
    'Medication': (read_medication_data, 100045),
    'Hearing' : (read_hearing_data, 100043),
    'Household' : (read_household_data, 100066),
    'MentalHealth' : (read_mental_health_data, 100060),
    'OtherSociodemographics' : (read_other_sociodemographics_data, 100067),
    'PhysicalActivity' : (read_physical_activity_data, 100054),
    'SexualFactors' : (read_sexual_factors_data, 100056),
    'Sleep' : (read_sleep_data, 100057),
    'SocialSupport' : (read_social_support_data, 100061),
    'SunExposure' : (read_sun_exposure_data, 100055)
}

medical_diagnoses_dict = dict(zip(['medical_diagnoses_%s' % letter for letter in ascii_uppercase], [(partial(read_medical_diagnoses_data, letter = letter), 41270) for letter in ascii_uppercase]))
map_envdataset_to_dataloader_and_field = {**map_envdataset_to_dataloader_and_field, **medical_diagnoses_dict}

## SUB LOADS : ETHNICITY, TARGET RESIDUALS, ENV DATA
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
        #df = load_data_env_(env_dataset, **kwargs)

        if env_dataset not in map_envdataset_to_dataloader_and_field.keys():
            raise ValueError('Wrong dataset name ! ')
        else :
            dataloader, field = map_envdataset_to_dataloader_and_field[env_dataset]
            df = dataloader(**kwargs)

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
            instance, field = dict_target_to_instance_and_id[target_dataset]
            df_organ['id'] = df_organ['eid'].astype(str) + '_' + str(instance)
            df_organ = df_organ.set_index('id')
    else :
        raise ValueError('')
    return df_organ[['residual', 'Sex', 'Age', 'eid']]


## FINAL LOAD




## NEED TO MOVE THIS !!

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
