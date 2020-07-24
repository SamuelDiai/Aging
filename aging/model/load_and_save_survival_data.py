import pandas as pd
import glob
from .load_and_save_data import load_data
from ..processing.base_processing import path_data
import numpy as np

path_features_survival = '/n/groups/patel/samuel/Survival/feature_importances/'
path_predictions_survival = '/n/groups/patel/samuel/Survival/predictions'
path_predictions_survival_regression = '/n/groups/patel/samuel/SurvivalRegression/predictions'

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


def read_death_record_updated(instances = [0, 1, 2, 3], **kwargs):
    path_death = '/n/groups/patel/uk_biobank/covid_death_071720/52887/death_071720.txt'
    path_death_cause = '/n/groups/patel/uk_biobank/covid_death_071720/52887/death_cause_071720.txt'
    df_sex_age = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv')
    list_df = []

    df_death = pd.read_csv(path_death, usecols = ['date_of_death', 'eid'], delim_whitespace= True)
    df_death = df_death.set_index('eid')
    df_death['Date of death'] = pd.to_datetime(df_death['date_of_death'])

    df_cause = pd.read_csv(path_death_cause, usecols = ['cause_icd10', 'eid'], delim_whitespace= True)
    df_cause = df_cause.set_index('eid')
    ## Remove accidents :
    df_cause = df_cause[~df_cause.cause_icd10.str.contains('V') & ~df_cause.cause_icd10.str.contains('W') & ~df_cause.cause_icd10.str.contains('X') & ~df_cause.cause_icd10.str.contains('Y')]
    def custom_apply(row):
        if row['cause_icd10'] <= 'D48' and row['cause_icd10'] >= 'C':
            return 'Cancer'
        elif row['cause_icd10'] <= 'I89' and row['cause_icd10'] >= 'I05':
            return 'CVD'
        else :
            return 'other'
    df_cause['Type of death'] = df_cause.apply(custom_apply, axis = 1)
    df_death_full = df_cause.join(df_death)

    for instance in instances:
        dict_rename = dict(zip(['eid', '53-%s.0' % instance], [ 'eid', 'Date of attending assessment centre']))
        df_attended = pd.read_csv(path_data, usecols = ['53-%s.0' % instance, 'eid'], **kwargs)
        df_attended = df_attended.rename(columns = dict_rename)
        df_attended['id'] = df_attended['eid'].astype(str) + '_%s' % instance
        df_attended = df_attended.set_index('id')

        list_df.append(df_attended)
    df = pd.concat(list_df)
    df = df.dropna()
    df = df.reset_index().merge(df_death_full.reset_index(), on = 'eid', how = 'outer').set_index('id')
    df['Date of last follow up'] = pd.to_datetime(df['Date of death'].max())
    def custom_apply(row):
        if not pd.isna(row['Date of death']):
            return (row['Date of death'] - row['Date of attending assessment centre']).days/365.25
        else :
            return (row['Date of last follow up'] - row['Date of attending assessment centre']).days/365.25
    df['Follow up time'] = df[['Date of last follow up', 'Date of death', 'Date of attending assessment centre']].apply(custom_apply, axis = 1)
    df['Is dead'] = ~df['Date of death'].isna()

    return df[['eid', 'Follow up time', 'Is dead', 'Date of death', 'Type of death']][~df['Follow up time'].isna()]


def load_data_survivalregression(dataset, target, **kwargs):
    df, organ, view = load_data(dataset, **kwargs)
    try :
        df_survival = pd.read_csv('/n/groups/patel/samuel/Survival/survival_updated.csv').set_index('id')
    except FileNotFoundError:
        df_survival = read_death_record_updated(**kwargs)
        df_survival.to_csv('/n/groups/patel/samuel/Survival/survival_updated.csv')
    df_survival = df_survival.drop(columns = ['eid'])
    df_full = df.join(df_survival)
    if target in ['CVD', 'Cancer']:
        df_full = df_full[df_full['Type of death'] == target]
    else :
        df_full = df_full[df_full['Is dead']].drop_duplicates('eid')

    df_full = df_full.drop(columns = ['Is dead', 'Date of death', 'Type of death']).drop_duplicates('eid')
    return df_full, organ, view

def load_data_survival(dataset, target, **kwargs):
    df, organ, view = load_data(dataset, **kwargs)
    try :
        df_survival = pd.read_csv('/n/groups/patel/samuel/Survival/survival_updated.csv').set_index('id')
    except FileNotFoundError:
        df_survival = read_death_record_updated(**kwargs)
        df_survival.to_csv('/n/groups/patel/samuel/Survival/survival_updated.csv')

    if target == 'CVD':
        df_survival = df_survival[df_survival['Type of death'].isin([np.nan, 'CVD'])]
    elif target == 'Cancer' :
        df_survival = df_survival[df_survival['Type of death'].isin([np.nan, 'Cancer'])]
    else :
        df_survival = df_survival[df_survival['Type of death'].isin([np.nan, 'other', 'Cancer', 'CVD'])]
    df_survival = df_survival.reset_index().drop_duplicates(['eid', 'id']).set_index('id')
    df_survival = df_survival.drop(columns = ['eid'])
    df_full = df.join(df_survival)
    df_full['y'] = list(zip(df_full['Is dead'], df_full['Follow up time']))
    df_full = df_full.drop(columns = ['Date of death', 'Type of death', 'Is dead', 'Follow up time'])
    return df_full, organ, view


def save_features_to_csv(cols, features_imp, organ, view, model_name, method):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    full_name = 'FeatureImpSurvival_'
    if method  == 'sd':
        full_name += 'sd_'
    elif method == 'mean':
        full_name += 'mean_'
    final_df.set_index('features').to_csv(path_features_survival + '/' + full_name + '_' + organ + '_' + view + '_' + model_name + '.csv')

def save_predictions_to_csv(predicts_df, step, target, dataset, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])
    filename = 'PredictionsSurvival_' + dataset + '_' +  target + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('id').to_csv(path_predictions_survival + '/' + filename)

def save_predictions_regression_to_csv(predicts_df, step, target, dataset, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])
    filename = 'PredictionsSurvivalRegression_' + dataset + '_' +  target + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('id').to_csv(path_predictions_survival_regression + '/' + filename)
