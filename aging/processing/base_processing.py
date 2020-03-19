import pandas as pd
import sys


# To edit for dev
if sys.platform == 'linux':
	path_data = "/n/groups/patel/uk_biobank/main_data_52887/ukb37397.csv"
	path_dictionary = "/n/groups/patel/samuel/HMS-Aging/Data_Dictionary_Showcase.csv"
	path_features = "/n/groups/patel/samuel/Aging/feature_importances/"
	path_predictions = "/n/groups/patel/samuel/Aging/predictions/"
elif sys.platform == 'darwin':
	path_data = "/Users/samuel/Desktop/ukbhead.csv"
	path_dictionary = "/Users/samuel/Downloads/drop/Data_Dictionary_Showcase.csv"
	path_features = "/Users/samuel/Desktop/Aging/feature_importances/"
	path_predictions = "/Users/samuel/Desktop/Aging/predictions/"


def read_data(cols_features, cols_filter, **kwargs):
	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']

	df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
	df_features.set_index('FieldID', inplace = True)
	feature_id_to_name = df_features.to_dict()['Field']


	temp = pd.read_csv(path_data, usecols = ['eid', '21003-2.0', '31-0.0'] + cols_features + cols_filter, nrows = nrows)
	temp.set_index('eid', inplace = True)

	## remove rows which contains any values for features in cols_features and then select only features in cols_abdominal
	if len(cols_filter) != 0:
		temp = temp[temp[cols_filter].isna().all(axis = 1)][['21003-2.0', '31-0.0'] + cols_features]
	else :
		temp = temp[['21003-2.0', '31-0.0'] + cols_features]
	## Remove rows which contains ANY Na

	features_index = temp.columns
	features = []
	for elem in features_index:
	    if elem != '21003-2.0' and elem != '31-0.0':
	        features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
	    else:
	        features.append(feature_id_to_name[int(elem.split('-')[0])])

	df = temp.dropna(how = 'any')
	df.columns = features
	return df


def read_and_save_data(cols_features, cols_filter, output_path, filename, **kwargs):
	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']

	df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
	df_features.set_index('FieldID', inplace = True)
	feature_id_to_name = df_features.to_dict()['Field']


	temp = pd.read_csv(path_data, usecols = ['eid', '21003-2.0', '31-0.0'] + cols_features + cols_filter, nrows = nrows)
	temp.set_index('eid', inplace = True)

	## remove rows which contains any values for features in cols_features and then select only features in cols_abdominal
	if len(cols_filter) != 0:
		temp = temp[temp[cols_filter].isna().all(axis = 1)][['21003-2.0', '31-0.0'] + cols_features]
	else :
		temp = temp[['21003-2.0', '31-0.0'] + cols_features]
	## Remove rows which contains ANY Na

	features_index = temp.columns
	features_index_clean = [int(elem.split('-')[0]) for elem in features_index]
	features = [feature_id_to_name[elem] for elem in features_index_clean]

	df = temp.dropna(how = 'any')
	df.columns = features
	df.to_csv(output_path + '/' + file_name + '.csv')
	return df
