import pandas as pd
from .base_processing import path_data, path_dictionary, read_data
"""
Features used :
	100014 - Autorefraction
	Errors features : None
	Missing : None
"""
def read_eye_data(**kwargs):
	cols_features =  ['20261-0.0', '5208-0.0', '5201-0.0'] + ['5264-0.0', '5256-0.0', '5265-0.0', '5257-0.0', '5262-0.0', '5254-0.0', '5263-0.0', '5255-0.0']
	cols_filter = []
	instance = 0
	a = read_data(cols_features, cols_filter, instance, **kwargs)
	b = read_eye_autorefraction_data(**kwargs)
	return a.join(b, rsuffix = '_del', lsuffix = '', how = 'inner').drop(columns = ['Age when attended assessment centre_del', 'Sex_del'])


def read_eye_acuity_data(**kwargs):
	cols_features =  ['20261-0.0', '5208-0.0', '5201-0.0']
	cols_filter = []
	instance = 0
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_eye_intraocular_pressure_data(**kwargs):
	cols_features =  ['5264-0.0', '5256-0.0', '5265-0.0', '5257-0.0', '5262-0.0', '5254-0.0', '5263-0.0', '5255-0.0']
	cols_filter = []
	instance = 0
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_eye_autorefraction_data(**kwargs):
	## deal with kwargs
	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']

	## Create feature name dict
	df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
	df_features.set_index('FieldID', inplace = True)
	feature_id_to_name = df_features.to_dict()['Field']

	## create features to extract for each category : left, right and 6mm, 3mm, keratometry
	l6 = sorted(['5110','5157','5113','5118','5162','5134','5102','5097']) # missing 5105
	r6 = sorted(['5109','5158','5114','5117','5161','5106','5133','5101','5098'])
	l3 = sorted(['5108','5159','5115','5116','5160','5107','5132','5100','5099'])
	r3 = sorted(['5111','5156','5112','5119','5163','5104','5135','5103','5096'])
	lg = sorted(['5089','5086','5085'])
	rg = sorted(['5088','5087','5084'])

	## index which corresponds to the best measuring process
	index_l3 = '5237-0.0'
	index_r3 = '5292-0.0'
	index_l6 = '5306-0.0'
	index_r6 = '5251-0.0'
	index_lg = '5276-0.0'
	index_rg = '5221-0.0'

	## read all the data

	temp = pd.read_csv(path_data,
	                   usecols = [elem + '-0.' + str(int_) for elem in r3 + l3 + r6 + l6 for int_ in range(6)]
	                             + [elem + '-0.' + str(int_) for elem in  lg + rg for int_ in range(10)]
	                             + [index_l3, index_r3, index_l6, index_r6, index_lg, index_rg]
	                             + ['eid', '21003-0.0', '31-0.0']
	                  ).set_index('eid')
	temp = temp[~temp[[index_r3, index_l3, index_r6, index_l6, index_lg, index_rg]].isna().any(axis = 1)]

	def apply_custom(row):
    #print(row)
	    index_l3 = int(row['5237-0.0'])
	    index_r3 = int(row['5292-0.0'])
	    index_l6 = int(row['5306-0.0'])
	    index_r6 = int(row['5251-0.0'])
	    index_lg = int(row['5276-0.0'])
	    index_rg = int(row['5221-0.0'])

	    arr_l3 = [str(elem) + '-0.' + str(index_l3) for elem in l3]
	    arr_r3 = [str(elem) + '-0.' + str(index_r3) for elem in r3]
	    arr_l6 = [str(elem) + '-0.' + str(index_l6) for elem in l6]
	    arr_r6 = [str(elem) + '-0.' + str(index_r6) for elem in r6]
	    arr_lg = [str(elem) + '-0.' + str(index_lg) for elem in lg]
	    arr_rg = [str(elem) + '-0.' + str(index_rg) for elem in rg]
	    return pd.Series(row[sorted(arr_l3 + arr_r3 + arr_l6 + arr_r6 + arr_lg + arr_rg + ['21003-0.0', '31-0.0'])].values)

	temp = temp.apply(apply_custom, axis = 1)
	temp.columns = sorted([elem + '-0.0' for elem in l3 + r3 + l6 + r6 + lg + rg] + ['21003-0.0', '31-0.0'])

	## Remova NAs
	df = temp[~temp.isna().any(axis = 1)]

	## Rename Columns
	features_index = temp.columns
	features = []
	for elem in features_index:
	    if elem != '21003-0.0' and elem != '31-0.0':
	        features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
	    else:
	        features.append(feature_id_to_name[int(elem.split('-')[0])])
	df.columns =  features
	return df
