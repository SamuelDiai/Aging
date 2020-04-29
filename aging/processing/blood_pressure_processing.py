from .base_processing import read_data, path_data, path_dictionary
import pandas as pd
"""
Features used :
	100011 - Blood Pressure
	Errors features : None
	Missing : None
"""

# def read_blood_pressure_data(**kwargs):
#     cols_features = ['102-0.0', '102-0.1', '4079-0.0', '4079-0.1', '4080-0.0', '4080-0.1']
#     cols_filter = []
#     instance = 0
#     temp = read_data(cols_features, cols_filter, instance, **kwargs)
#     for elem in pd.Series([elem.split('.')[0] for elem in temp.columns.values if 'Age' not in elem and 'Sex' not in elem]).drop_duplicates():
#         temp[elem + '.0'] = (temp[elem + '.0'] + temp[elem + '.1'])/2
#     return temp[[elem for elem in temp.columns if '.1' not in elem]]




def read_blood_pressure_data(**kwargs):
	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']

	df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
	df_features.set_index('FieldID', inplace = True)
	feature_id_to_name = df_features.to_dict()['Field']


	instances = [0, 1, 2, 3]
	list_df = []
	for instance in instances :
		age_col = '21003-' + str(instance) + '.0'
		cols_features_ = ['102-%s.0' % instance, '102-%s.1' % instance, '4079-%s.0' % instance, '4079-%s.1' % instance, '4080-%s.0' % instance, '4080-%s.1' % instance]
		temp = pd.read_csv(path_data, usecols = ['eid', age_col, '31-0.0'] + cols_features_, nrows = nrows)
		temp.set_index('eid', inplace = True)
		temp.index = temp.index.rename('id')

		## remove rows which contains any values for features in cols_features and then select only features in cols_abdominal
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


		for elem in pd.Series([elem.split('.')[0] for elem in df.columns.values if 'Age' not in elem and 'Sex' not in elem]).drop_duplicates():
			df[elem + '.0'] = (df[elem + '.0'] + df[elem + '.1'])/2
		df = df[[elem for elem in df.columns if '.1' not in elem]]
		df['eid'] = df.index
		df.index = df.index.astype('str') + '_' + str(instance)
		list_df.append(df)

	return pd.concat(list_df)
