import pandas as pd
import sys


# To edit for dev
if sys.platform == 'linux':
	path_data = "/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv"
	path_dictionary = "/n/groups/patel/samuel/HMS-Aging/Data_Dictionary_Showcase.csv"
	path_features = "/n/groups/patel/samuel/feature_importances4/"
	path_predictions = "/n/groups/patel/samuel/predictions4/"
	path_inputs = "/n/groups/patel/samuel/inputs_v2/"
elif sys.platform == 'darwin':
	path_data = "/Users/samuel/Desktop/ukbhead.csv"
	path_dictionary = "/Users/samuel/Downloads/drop/Data_Dictionary_Showcase.csv"
	path_features = "/Users/samuel/Desktop/Aging/feature_importances/"
	#path_predictions = "/Users/samuel/Desktop/Aging/predictions/"
	path_predictions = "/Users/samuel/Desktop/Aging/predictions2/"
	path_inputs = "/Users/samuel/Desktop/Aging/inputs/"


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
        temp = pd.read_csv(path_data, usecols = ['eid', age_col, '31-0.0'] + cols_features_ + cols_filter, nrows = nrows)
        temp.set_index('eid', inplace = True)
        temp.index = temp.index.rename('id')

        ## remove rows which contains any values for features in cols_features and then select only features in cols_abdominal
        if len(cols_filter) != 0:
            temp = temp[temp[cols_filter].isna().all(axis = 1)][[age_col, '31-0.0'] + cols_features_]
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

        df = temp.dropna(how = 'any')

        df.columns = features
        df['eid'] = df.index
        df.index = df.index.astype('str') + '_' + str(instance)
        list_df.append(df)
        print(df)
    return pd.concat(list_df)

def read_data_and_merge_temporal_features(cols_features, timesteps, instance,  **kwargs):
	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']

	df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
	df_features.set_index('FieldID', inplace = True)
	feature_id_to_name = df_features.to_dict()['Field']
	list_df = []
	for instance_ in instance:
		age_col = '21003-' + str(instance) + '.0'

		multi_cols = [str(elem) + '-%s.' % instance + str(int_) for elem in cols_features for int_ in range(timesteps)] + ['eid', age_col, '31-0.0']
		big_df = pd.read_csv(path_data, usecols = multi_cols)
		dict_data = {}
		dict_data['eid'] = big_df['eid']
		dict_data[age_col] = big_df[age_col]
		dict_data['31-0.0'] = big_df['31-0.0']
		for elem in cols_features :
		    dict_data[str(elem) + '-2.0'] = big_df[[str(elem) + '-%s.' % instance + str(int_) for int_ in range(timesteps)]].mean(axis = 1).values

		temp = pd.DataFrame(data = dict_data).set_index('eid')
		features_index = temp.columns
		features = []
		for elem in features_index:
		    if elem != age_col and elem != '31-0.0':
		        features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
		    else:
		        features.append(feature_id_to_name[int(elem.split('-')[0])])

		df = temp.dropna(how = 'any')
		df.columns = features

		df['eid'] = df.index
		df.index = df.index.astype('str') + '_' + str(instance)
		list_df.append(df)
	return pd.concatenate(list_df)
