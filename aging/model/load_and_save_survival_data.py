import pandas as pd
import glob
from .load_and_save_data import load_data
from ..processing.base_processing import path_data

path_features_survival = '/n/groups/patel/samuel/Survival/feature_importances/'
path_predictions_survival = '/n/groups/patel/samuel/Survival/predictions'

def read_death_record(instances = [0, 1, 2, 3], **kwargs):
    df_sex_age = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv')
    list_df = []
    for instance in instances:
        dict_rename = dict(zip(['40000-0.0', '40007-0.0', 'eid', '53-%s.0' % instance], ['Date of death', 'Age at death', 'eid', 'Date of attending assessment centre']))
        df = pd.read_csv(path_data, usecols = ['40000-0.0', '40007-0.0', 'eid', '53-%s.0' % instance], **kwargs)
        df = df.rename(columns = dict_rename)
        df['id'] = df['eid'].astype(str) + '_%s' % instance
        df = df.set_index('id')
        df = df[~df['Date of attending assessment centre'].isna()]
        df['Date of death'] = pd.to_datetime(df['Date of death'])
        df['Date of attending assessment centre'] = pd.to_datetime(df['Date of attending assessment centre'])

        list_df.append(df)

    df = pd.concat(list_df)
    df['Date of last follow up'] = pd.to_datetime(df['Date of attending assessment centre'].max())
    def custom_apply(row):
        if not pd.isna(row['Date of death']):
            return (row['Date of death'] - row['Date of attending assessment centre']).days/365.25
        else :
            return (row['Date of last follow up'] - row['Date of attending assessment centre']).days/365.25
    df['Follow up time'] = df[['Date of last follow up', 'Date of death', 'Date of attending assessment centre']].apply(custom_apply, axis = 1)
    df['Is dead'] = ~df['Age at death'].isna()
    return df[['Follow up time', 'Is dead']]

def load_data_survival(dataset, **kwargs):
    df, organ, view = load_data(dataset, **kwargs)
    try :
        df_survival = pd.read_csv('/n/groups/patel/samuel/Survival/survival.csv').set_index('id')
    except FileNotFoundError:
        df_survival = read_death_record(**kwargs)
        df_survival.to_csv('/n/groups/patel/samuel/Survival/survival.csv')
    df_full = df.join(df_survival)
    df_full['y'] = list(zip(df_full['Is dead'], df_full['Follow up time']))
    df_full = df_full.drop(columns = ['Is dead', 'Follow up time'])
    return df_full, organ, view


def save_features_to_csv(cols, features_imp, organ, view, model_name, method):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    full_name = 'FeatureImpSurvival_'
    if method  == 'sd':
        full_name += 'sd_'
    elif method == 'mean':
        full_name += 'mean_'
    final_df.set_index('features').to_csv(path_features_survival + '/' + full_name + '_' + organ + '_' + view + '_' + model_name + '.csv')

def save_predictions_to_csv(predicts_df, step, dataset, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])
    try :
        field, dataloader = map_dataset_to_field_and_dataloader[dataset]
    except KeyError:
        field = 'Cluster'
    filename = 'PredictionsSurvival_' + '_' + dataset + '_' + str(field) + '_main' +  '_raw' + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('id').to_csv(path_predictions_survival + '/' + filename)
