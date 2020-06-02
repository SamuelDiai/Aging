import pandas as pd
import sys


# To edit for dev
if sys.platform == 'linux':
	path_data = "/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv"
	path_dictionary = "/n/groups/patel/samuel/HMS-Aging/Data_Dictionary_Showcase.csv"
	path_features = "/n/groups/patel/samuel/feature_importances_final/"
	path_predictions = "/n/groups/patel/samuel/predictions_final/"
	path_inputs = "/n/groups/patel/samuel/final_inputs/"
elif sys.platform == 'darwin':
	path_data = "/Users/samuel/Desktop/ukbhead.csv"
	path_dictionary = "/Users/samuel/Downloads/drop/Data_Dictionary_Showcase.csv"
	path_features = "/Users/samuel/Desktop/Aging/feature_importances/"
	#path_predictions = "/Users/samuel/Desktop/Aging/predictions/"
	path_predictions = "/Users/samuel/Desktop/Aging/predictions2/"
	path_inputs = "/Users/samuel/Desktop/Aging/inputs/"


def read_ethnicity_data(**kwargs):
    dict_ethnicity_codes = {'1': 'White', '1001': 'British', '1002': 'Irish',
                                '1003': 'White_Other',
                                '2': 'Mixed', '2001': 'White_and_Black_Caribbean', '2002': 'White_and_Black_African',
                                '2003': 'White_and_Asian', '2004': 'Mixed_Other',
                                '3': 'Asian', '3001': 'Indian', '3002': 'Pakistani', '3003': 'Bangladeshi',
                                '3004': 'Asian_Other',
                                '4': 'Black', '4001': 'Caribbean', '4002': 'African', '4003': 'Black_Other',
                                '5': 'Chinese',
                                '6': 'Other_ethnicity',
                                '-1': 'Do_not_know',
                                '-3': 'Prefer_not_to_answer',
                                '-5': 'NA'}
    df = pd.read_csv(path_data, usecols = ['21000-0.0', '21000-1.0', '21000-2.0', 'eid'], **kwargs).set_index('eid')
    df.columns = ['Ethnicity', 'Ethnicity_1', 'Ethnicity_2']

    eids_missing_ethnicity = df.index[df['Ethnicity'].isna()]
    #print(eids_missing_ethnicity)
    for eid in eids_missing_ethnicity:
        sample = df.loc[eid, :]
        if not pd.isna(sample['Ethnicity_1']):
            df.loc[eid, 'Ethnicity'] = df.loc[eid, 'Ethnicity_1']
        elif not pd.isna(sample['Ethnicity_2']):
            df.loc[eid, 'Ethnicity'] = df.loc[eid, 'Ethnicity_2']
    df.drop(['Ethnicity_1', 'Ethnicity_2'], axis=1, inplace=True)
    df['Ethnicity'] = df['Ethnicity'].fillna(-5).astype(int).astype(str)

    #display(df)
    ethnicities = pd.get_dummies(df['Ethnicity'])
    ethnicities.rename(columns=dict_ethnicity_codes, inplace=True)
    ethnicities['White'] = ethnicities['White'] + ethnicities['British'] + ethnicities['Irish'] + ethnicities['White_Other']
    ethnicities['Mixed'] = ethnicities['Mixed'] + ethnicities['White_and_Black_Caribbean'] + ethnicities['White_and_Black_African'] + ethnicities['White_and_Asian'] + \
                            ethnicities['Mixed_Other']
    ethnicities['Asian'] = ethnicities['Asian'] + ethnicities['Indian'] + ethnicities['Pakistani'] + \
                           ethnicities['Bangladeshi'] + ethnicities['Asian_Other']
    ethnicities['Black'] = ethnicities['Black'] + ethnicities['Caribbean'] + ethnicities['African'] + \
                           ethnicities['Black_Other']
    ethnicities['Other'] = ethnicities['Other_ethnicity'] + ethnicities['Do_not_know'] + \
                           + ethnicities['Prefer_not_to_answer'] + ethnicities['NA']
    return ethnicities




def read_data(cols_features, cols_filter, instances, **kwargs):
    nrows = None
    if 'nrows' in kwargs.keys():
        nrows = kwargs['nrows']

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']

    if isinstance(instances, int):
        instances = [instances]

    list_df = []
    for instance in instances :
        age_col = '21003-' + str(instance) + '.0'
        cols_features_ = [str(feature) + '-%s.0' % instance for feature in cols_features]
        cols_filter_ = [str(filter) + '-%s.0' % instance for filter in cols_filter]
        temp = pd.read_csv(path_data, usecols = ['eid', age_col, '31-0.0'] + cols_features_ + cols_filter_, nrows = nrows)
        temp.set_index('eid', inplace = True)
        temp.index = temp.index.rename('id')

        ## remove rows which contains any values for features in cols_features and then select only features in cols_abdominal
        if len(cols_filter_) != 0:
            temp = temp[temp[cols_filter_].isna().all(axis = 1)][[age_col, '31-0.0'] + cols_features_]
        else :
            temp = temp[[age_col, '31-0.0'] + cols_features_]
        ## Remove rows which contains ANY Na

        features_index = temp.columns
        features = []
        for elem in features_index:
            if elem != age_col and elem != '31-0.0':
                features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
            else:
                features.append(feature_id_to_name[int(elem.split('-')[0])])

        df = temp[~temp[cols_features_].isna().all(axis = 1)]
        #df = temp.dropna(how = 'any')


        df.columns = features
        df['eid'] = df.index
        df.index = df.index.astype('str') + '_' + str(instance)
        list_df.append(df)
        #print(df)
    return pd.concat(list_df)

def read_data_and_merge_temporal_features(cols_features, timesteps, instance,  **kwargs):
    nrows = None
    if 'nrows' in kwargs.keys():
        nrows = kwargs['nrows']

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']
    list_features = []
    list_df = []
    for instance_ in instance:
        age_col = '21003-' + str(instance_) + '.0'

        multi_cols = [str(elem) + '-%s.' % instance_ + str(int_) for elem in cols_features for int_ in range(timesteps)] + ['eid', age_col, '31-0.0']
        big_df = pd.read_csv(path_data, usecols = multi_cols, nrows = nrows)
        dict_data = {}
        dict_data['eid'] = big_df['eid']
        dict_data[age_col] = big_df[age_col]
        dict_data['31-0.0'] = big_df['31-0.0']
        for elem in cols_features :
            dict_data[str(elem) + '-2.0'] = big_df[[str(elem) + '-%s.' % instance_ + str(int_) for int_ in range(timesteps)]].mean(axis = 1).values
            list_features.append(str(elem) + '-2.0')
        temp = pd.DataFrame(data = dict_data).set_index('eid')
        temp.index = temp.index.rename('id')
        features_index = temp.columns
        features = []
        for elem in features_index:
            if elem != age_col and elem != '31-0.0':
                features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
            else:
                features.append(feature_id_to_name[int(elem.split('-')[0])])

        df = temp[~temp[list_features].isna().all(axis = 1)]
        #df = temp.dropna(how = 'all')
        df.columns = features

        df['eid'] = df.index
        df.index = df.index.astype('str') + '_' + str(instance_)
        list_df.append(df)
    return pd.concat(list_df)
