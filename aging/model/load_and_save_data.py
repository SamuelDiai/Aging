import pandas as pd
import glob
from ..processing.base_processing import path_features , path_predictions, path_inputs, read_ethnicity_data
from ..processing.abdominal_composition_processing import read_abdominal_data
from ..processing.brain_processing import read_grey_matter_volumes_data, read_subcortical_volumes_data, read_brain_data, read_brain_dMRI_weighted_means_data
from ..processing.heart_processing import read_heart_data, read_heart_size_data, read_heart_PWA_data
from ..processing.body_composition_processing import read_body_composition_data
from ..processing.bone_composition_processing import read_bone_composition_data
from ..processing.ecg_processing import read_ecg_at_rest_data
from ..processing.anthropometry_processing import read_anthropometry_impedance_data, read_anthropometry_body_size_data, read_anthropometry_data
from ..processing.biochemestry_processing import read_blood_biomarkers_data, read_urine_biomarkers_data, read_blood_count_data, read_blood_data, read_urine_and_blood_data
from ..processing.eye_processing import read_eye_autorefraction_data, read_eye_acuity_data, read_eye_intraocular_pressure_data, read_eye_data
from ..processing.spirometry_processing import read_spirometry_data
from ..processing.blood_pressure_processing import read_blood_pressure_data
from ..processing.arterial_stiffness_processing import read_arterial_stiffness_data
from ..processing.mix_processing import read_arterial_and_bp_data, read_spiro_and_arterial_and_bp_data
from ..processing.carotid_ultrasound_processing import read_carotid_ultrasound_data
from ..processing.bone_densitometry_processing import read_bone_densitometry_data
from ..processing.hand_grip_strength_processing import read_hand_grip_strength_data
from ..processing.hearing_tests_processing import read_hearing_test_data
from ..processing.cognitive_tests_processing import read_reaction_time_data, read_matrix_pattern_completion, read_tower_rearranging_data, \
                                                    read_symbol_digit_substitution_data, read_paired_associative_learning_data, \
                                                    read_prospective_memory_data, read_numeric_memory_data, read_fluid_intelligence_data



map_dataset_to_field_and_dataloader = {
                    'AbdominalComposition' : (149, read_abdominal_data),
                    'BrainGreyMatterVolumes' : (1101, read_grey_matter_volumes_data),
                    'BrainSubcorticalVolumes': (1102, read_subcortical_volumes_data),
                    'BrainAllBiomarkers' : (100, read_brain_data),
                    'HeartAllBiomarkers' : (102, read_heart_data),
                    'HeartSize' : (133, read_heart_size_data),
                    'HeartPWA' : (128, read_heart_PWA_data),
                    'BodyComposition' : (124, read_body_composition_data),
                    'BoneComposition' : (125, read_bone_composition_data),
                    'ECGAtRest' : (12657, read_ecg_at_rest_data),
                    'AnthropometryImpedance' : (100008, read_anthropometry_impedance_data),
                    'UrineBiochemestry' : (100083, read_urine_biomarkers_data),
                    'BloodBiochemestry' : (17518, read_blood_biomarkers_data),
                    'BloodCount' : (100081, read_blood_count_data),  # Need to do blood infection
                    'Blood' : (100080, read_blood_data),
                    'UrineAndBlood' : ('Custom', read_urine_and_blood_data),
                    'EyeAutorefraction' : (100014, read_eye_autorefraction_data),
                    'EyeAcuity' : (100017, read_eye_acuity_data),
                    'EyeIntraoculaPressure' : (100015, read_eye_intraocular_pressure_data),
                    'EyesAllBiomarkers' : (100013, read_eye_data),
                    'BraindMRIWeightedMeans' : (135, read_brain_dMRI_weighted_means_data),
                    'Spirometry' :  (100020, read_spirometry_data),
                    'BloodPressure' : (100011, read_blood_pressure_data),
                    'AnthropometryBodySize' : (100010, read_anthropometry_body_size_data),
                    'AnthropometryAllBiomarkers' : (100008, read_anthropometry_data),
                    'ArterialStiffness' : (100007, read_arterial_stiffness_data),
                    'ArterialAndBloodPressure' : ('Custom', read_arterial_and_bp_data),
                    'SpiroAndArterialAndBp' : ('Custom', read_spiro_and_arterial_and_bp_data),
                    'CarotidUltrasound' : (101, read_carotid_ultrasound_data),
                    'BoneDensitometryOfHeel' : (100018, read_bone_densitometry_data),
                    'HandGripStrength' : (100019, read_hand_grip_strength_data),
                    'HearingTest' : (100049, read_hearing_test_data),
                    'CognitiveReactionTime' : (100032, read_reaction_time_data),
                    'CognitiveMatrixPatternCompletion' : (501, read_matrix_pattern_completion),
                    'CognitiveTowerRearranging' : (503, read_tower_rearranging_data),
                    'CognitiveSymbolDigitSubstitution' : (502, read_symbol_digit_substitution_data),
                    'CognitivePairedAssociativeLearning' : (506, read_paired_associative_learning_data),
                    'CognitiveProspectiveMemory' : (100031, read_prospective_memory_data),
                    'CognitiveNumericMemory' : (100029, read_numeric_memory_data),
                    'CognitiveFluidIntelligence' : (100027, read_fluid_intelligence_data)
                    }


dict_dataset_to_organ_and_view = {
    'BrainGreyMatterVolumes' : ('Brain', 'GreyMatterVolumes'),
    'BrainSubcorticalVolumes': ('Brain', 'SubcorticalVolumes'),
    'BrainAllBiomarkers' : ('Brain', 'AllBiomarkers'),
    'HeartAllBiomarkers' : ('Heart', 'HeartAllBiomarkers'),
    'HeartSize' : ('Heart', 'Size'),
    'HeartPWA' : ('Heart', 'PWA'),
    'BodyComposition' : ('BodyComposition', 'main'),
    'BoneComposition' : ('BoneComposition', 'main'),
    'ECGAtRest' : ('ECG', 'BiomarkersAtRest'),
    'AnthropometryImpedance' : ('Anthropometry', 'Impedance'),
    'UrineBiochemestry' : ('Urine', 'Biochemistry'),
    'BloodBiochemestry' : ('BloodB', 'BloodBiochemistry'),
    'BloodCount' : ('BloodC', 'BloodCount'),  # Need to do blood infection
    'EyeAutorefraction' : ('Eyes', 'Autorefraction'),
    'EyeAcuity' : ('Eyes', 'Acuity'),
    'EyeIntraoculaPressure' : ('Eyes', 'IntraocularPressure'),
    'EyesAllBiomarkers' : ('Eyes', 'AllBiomarkers'),
    'BraindMRIWeightedMeans' : ('Brain', 'dMRIWeightedMeans'),
    'Spirometry' :  ('Lungs', 'Spirometry'),
    'BloodPressure' : ('BloodPressure', 'Biomarkers'),
    'AnthropometryBodySize' : ('Anthropometry', 'BodySize'),
    'AnthropometryAllBiomarkers' : ('Anthropometry', 'AllBiomarkers'),
    'ArterialStiffness' : ('Vascular', 'BiomarkersArterialStiffness'),
    'CarotidUltrasound' : ('Carotids', 'BiomarkersUltrasound'),
    'BoneDensitometryOfHeel' : ('Heel', 'BoneDensitometry'),
    'HandGripStrength' : ('Hand', 'GripStrenght'),
    'HearingTest' : ('Hearing', 'HearingTest'),
    'CognitiveReactionTime' : ('Cognitive', 'ReactionTime'),
    'CognitiveMatrixPatternCompletion' : ('Cognitive', 'MatrixPatternCompletion'),
    'CognitiveTowerRearranging' : ('Cognitive', 'TowerRearranging'),
    'CognitiveSymbolDigitSubstitution' : ('Cognitive', 'SymbolDigitSubstitution'),
    'CognitivePairedAssociativeLearning' : ('Cognitive', 'PairedAssociativeLearning'),
    'CognitiveProspectiveMemory' : ('Cognitive', 'ProspectiveMemory'),
    'CognitiveNumericMemory' : ('Cognitive', 'NumericMemory'),
    'CognitiveFluidIntelligence' : ('Cognitive', 'FluidIntelligence')
}

# def load_data(dataset, **kwargs):
#     selected_inputs = glob.glob(path_inputs + '%s.csv' % dataset)
#     print(selected_inputs)
#     if len(selected_inputs) == 0:
#         print("Load New Data")
#         #df = load_data_(dataset, **kwargs)
#         if dataset not in map_dataset_to_field_and_dataloader.keys():
#             raise ValueError('Wrong dataset name ! ')
#         else :
#             field, dataloader = map_dataset_to_field_and_dataloader[dataset]
#             df = dataloader(**kwargs)
#         df.to_csv(path_inputs + dataset + '.csv')
#         return df.dropna()
#     elif len(selected_inputs) == 1 :
#         nrows = None
#         if 'nrows' in kwargs.keys():
#             nrows = kwargs['nrows']
#         print("Load Existing Data")
#         df = pd.read_csv(selected_inputs[0], nrows = nrows).set_index('id')
#         return df.dropna()
#     else :
#         print("Error")
#         raise ValueError('Too many Input file for the selected dataset')

def load_data(dataset, **kwargs):
    if 'Cluster' in dataset :
        df = pd.read_csv(dataset).set_index('id')
        organ, view = 'Cluster', 'main'
    elif '/n' not in dataset:
        path_dataset = path_inputs + dataset + '.csv'
        df = pd.read_csv(path_dataset).set_index('id')
        df_ethnicity_sex_age = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv').set_index('id')
        df = df_ethnicity_sex_age.join(df, rsuffix = '_r')
        df = df[df.columns[~df.columns.str.contains('_r')]]
        organ, view = dict_dataset_to_organ_and_view[dataset]
    return df.dropna(), organ, view

def create_data(dataset, **kwargs):
    if dataset not in map_dataset_to_field_and_dataloader.keys():
        raise ValueError('Wrong dataset name ! ')
    else :
        field, dataloader = map_dataset_to_field_and_dataloader[dataset]
        df = dataloader(**kwargs)
    df.to_csv(path_inputs + dataset + '.csv')

def save_features_to_csv(cols, features_imp, target, dataset, model_name, sd):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    full_name = 'FeatureImp_'
    if sd :
        full_name += 'sd_'
    final_df.set_index('features').to_csv(path_features + '/' + full_name + target + '_' + dataset + '_' + model_name + '.csv')

def save_predictions_to_csv(predicts_df, step, target, dataset, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

    try :
        field, dataloader = map_dataset_to_field_and_dataloader[dataset]
    except KeyError:
        field = 'Cluster'
    filename = 'Predictions_' + target + '_' + dataset + '_' + str(field) + '_main' +  '_raw' + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('id').to_csv(path_predictions + '/' + filename)
