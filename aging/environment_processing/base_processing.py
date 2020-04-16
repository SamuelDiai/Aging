import pandas as pd
import sys

# To edit for dev
if sys.platform == 'linux':
	path_data = "/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv"
	path_dictionary = "/n/groups/patel/samuel/HMS-Aging/Data_Dictionary_Showcase.csv"
	path_features = "/n/groups/patel/samuel/EWAS/feature_importances/"
	path_predictions = "/n/groups/patel/samuel/EWAS/predictions/"
	path_inputs_env = "/n/groups/patel/samuel/EWAS/inputs/"
	path_target_residuals = "/n/groups/patel/samuel/residuals/"
elif sys.platform == 'darwin':
	path_data = "/Users/samuel/Desktop/ukbhead.csv"
	path_dictionary = "/Users/samuel/Downloads/drop/Data_Dictionary_Showcase.csv"
	path_features = "/Users/samuel/Desktop/EWAS/feature_importances/"
	path_predictions = "/Users/samuel/Desktop/EWAS/predictions/"
	path_inputs_env = "/n/groups/patel/samuel/EWAS/inputs/"
	path_target_residuals = "/n/groups/patel/samuel/residuals/"


def read_data(cols_categorical, cols_features, instance, **kwargs):
    nrows = None
    if 'nrows' in kwargs.keys():
        nrows = kwargs['nrows']


    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']

    #age_col = '21003-' + str(instance) + '.0'
    #print("PATH DATA : ", path_data)
    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_features + cols_categorical, nrows = nrows)
    for col in cols_categorical:
        temp[col] = temp[col].astype('Int64')
    temp.set_index('eid', inplace = True)

    features_index = temp.columns
    features = []
    for elem in features_index:
        features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
    df = temp.dropna(how = 'any')
    df.columns = features
    return df
