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



map_dataset_to_field_and_dataloader = {
                    'AbdominalComposition' : (149, read_abdominal_data),
                    'BrainGreyMatterVolumes' : (1101, read_grey_matter_volumes_data),
                    'BrainSubcorticalVolumes': (1102, read_subcortical_volumes_data),
                    'Brain' : (100, read_brain_data),
                    'Heart' : (102, read_heart_data),
                    'HeartSize' : (133, read_heart_size_data),
                    'HeartPWA' : (128, read_heart_PWA_data),
                    'BodyComposition' : (124, read_body_composition_data),
                    'BoneComposition' : (125, read_bone_composition_data),
                    'ECGAtRest' : (12657, read_ecg_at_rest_data),
                    'AnthropometryImpedance' : (100008, read_anthropometry_data),
                    'UrineBiochemestry' : (100083, read_urine_biomarkers_data),
                    'BloodBiochemestry' : (17518, read_blood_biomarkers_data),
                    'BloodCount' : (100081, read_blood_count_data),  # Need to do blood infection
                    'Blood' : (100080, read_blood_data),
                    'UrineAndBlood' : ('Custom', read_urine_and_blood_data),
                    'EyeAutorefraction' : (100014, read_eye_autorefraction_data),
                    'EyeAcuity' : (100017, read_eye_acuity_data),
                    'EyeIntraoculaPressure' : (100015, read_eye_intraocular_pressure_data),
                    'Eye' : (100013, read_eye_data),
                    'BraindMRIWeightedMeans' : (135, read_brain_dMRI_weighted_means_data),
                    'Spirometry' :  (100020, read_spirometry_data),
                    'BloodPressure' : (100011, read_blood_pressure_data),
                    'AnthropometryBodySize' : (100010, read_anthropometry_body_size_data),
                    'Anthropometry' : (100008, read_anthropometry_data),
                    'ArterialStiffness' : (100007, read_arterial_stiffness_data),
                    'ArterialAndBloodPressure' : ('Custom', read_arterial_and_bp_data),
                    'SpiroAndArterialAndBp' : ('Custom', read_spiro_and_arterial_and_bp_data),
                    'CarotidUltrasound' : (101, read_carotid_ultrasound_data),
                    }

class DataLoader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = None
        self.path_inputs = path_inputs

    def load_data(self, **kwargs):
        selected_inputs = glob.glob(self.path_inputs + '%s.csv' % self.dataset)
        print(selected_inputs)
        if len(selected_inputs) == 0:
            print("Load New Data")
            #df = load_data_(self.dataset, **kwargs)

            if self.dataset not in map_dataset_to_field_and_dataloader.keys():
                raise ValueError('Wrong dataset name ! ')
            else :
                field, dataloader = map_dataset_to_field_and_dataloader[self.dataset]
                df = dataloader(**kwargs)
            df.to_csv(self.path_inputs + self.dataset + '.csv')
            self.df = df
        elif len(selected_inputs) == 1 :
            nrows = None
            if 'nrows' in kwargs.keys():
                nrows = kwargs['nrows']
            print("Load Existing Data")
            df = pd.read_csv(selected_inputs[0], nrows = nrows).set_index('id')
            self.df = df
        else :
            print("Error")
            raise ValueError('Too many Input file for the selected dataset')


    # def load_data_(self, **kwargs):
    #     nrows = None
    #     dataset = self.dataset
    #     if 'nrows' in kwargs.keys():
    #         nrows = kwargs['nrows']
    #     if dataset not in dataset_to_field.keys():
    #         raise ValueError('Wrong dataset name ! ')
    #     else :
    #         if dataset == 'AbdominalComposition':
    #             df = read_abdominal_data(**kwargs)
    #         elif dataset == 'Brain':
    #             df = read_brain_data(**kwargs)
    #         elif dataset == 'BrainGreyMatterVolumes':
    #             df = read_grey_matter_volumes_data(**kwargs)
    #         elif dataset == 'BrainSubcorticalVolumes':
    #             df = read_subcortical_volumes_data(**kwargs)
    #         elif dataset == 'Heart':
    #             df = read_heart_data(**kwargs)
    #         elif dataset == 'HeartSize':
    #             df = read_heart_size_data(**kwargs)
    #         elif dataset == 'HeartPWA':
    #             df = read_heart_PWA_data(**kwargs)
    #         elif dataset == 'BoneComposition':
    #             df = read_bone_composition_data(**kwargs)
    #         elif dataset == 'BodyComposition':
    #             df = read_body_composition_data(**kwargs)
    #         elif dataset == 'ECGAtRest':
    #             df = read_ecg_at_rest_data(**kwargs)
    #         elif dataset == 'AnthropometryImpedance':
    #             df = read_anthropometry_impedance_data(**kwargs)
    #         elif dataset == 'UrineBiochemestry':
    #             df = read_urine_biomarkers_data(**kwargs)
    #         elif dataset == 'BloodBiochemestry':
    #             df = read_blood_biomarkers_data(**kwargs)
    #         elif dataset == 'BloodCount':
    #             df = read_blood_count_data(**kwargs)
    #         elif dataset == 'Blood':
    #             df = read_blood_data(**kwargs)
    #         elif dataset == 'UrineAndBlood':
    #             df = read_urine_and_blood_data(**kwargs)
    #         elif dataset == 'EyeAutorefraction':
    #             df = read_eye_autorefraction_data(**kwargs)
    #         elif dataset == 'EyeIntraoculaPressure':
    #             df = read_eye_intraocular_pressure_data(**kwargs)
    #         elif dataset == 'EyeAcuity':
    #             df = read_eye_acuity_data(**kwargs)
    #         elif dataset == 'Eye':
    #             df = read_eye_data(**kwargs)
    #         elif dataset == 'BraindMRIWeightedMeans':
    #             df = read_brain_dMRI_weighted_means_data(**kwargs)
    #         elif dataset == 'BloodPressure':
    #             df = read_blood_pressure_data(**kwargs)
    #         elif dataset == 'Spirometry':
    #             df = read_spirometry_data(**kwargs)
    #         elif dataset == 'AnthropometryBodySize':
    #             df = read_anthropometry_body_size_data(**kwargs)
    #         elif dataset == 'Anthropometry':
    #             df = read_anthropometry_data(**kwargs)
    #         elif dataset == 'ArterialStiffness':
    #             df = read_arterial_stiffness_data(**kwargs)
    #         elif dataset == 'ArterialAndBloodPressure':
    #             df = read_arterial_and_bp_data(**kwargs)
    #         elif dataset == 'SpiroAndArterialAndBp':
    #             df = read_spiro_and_arterial_and_bp_data(**kwargs)
    #         elif dataset == 'CarotidUltrasound':
    #             df = read_carotid_ultrasound_data(**kwargs)
    #
    #         return df




def save_features_to_csv(cols, features_imp, target, dataset, model_name):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    final_df.set_index('features').to_csv(path_features + '/' + 'FeatureImp_' + target + '_' + dataset + '_' + model_name + '.csv')


def save_predictions_to_csv(predicts_df, step, target, dataset, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

    filename = 'Predictions_' + target + '_' + dataset + '_' + str(dataset_to_field[dataset]) + '_main' +  '_raw' + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('id').to_csv(path_predictions + '/' + filename)
