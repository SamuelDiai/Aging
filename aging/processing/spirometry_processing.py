from .base_processing import path_data, path_dictionary

"""
Features used :
	100020 - FVC, FEV1, PEF
	Errors features : None
	Missing : None
"""

def read_spirometry_data(**kwargs):
    ## deal with kwargs
    nrows = None
    if 'nrows' in kwargs.keys():
        nrows = kwargs['nrows']

    ## Create feature name dict
    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']

    def custom_apply(row):
        flag_0 = row['3061-0.0']
        flag_1 = row['3061-0.1']
        flag_2 = row['3061-0.2']
        if flag_0 == 0 or flag_0 == 32:
            return pd.Series(row[['3064-0.0', '3062-0.0', '3063-0.0'] + ['21003-0.0', '31-0.0']].values)
        else:
            if flag_1 == 0 or flag_1 == 32:
                return pd.Series(row[['3064-0.1', '3062-0.1', '3063-0.1'] + ['21003-0.0', '31-0.0']].values)
            else :
                if flag_2 == 0 or flag_2 == 32:
                    return pd.Series(row[['3064-0.2', '3062-0.2', '3063-0.2'] + ['21003-0.0', '31-0.0']].values)
                else:
                    return  pd.Series([None, None, None] + [None, None])



    cols = ['3064-0.', '3062-0.', '3063-0.', '3061-0.']
    temp = pd.read_csv(path_data,  nrows = nrows, usecols = ['20151-0.0'] + [elem + str(int_) for elem in cols for int_ in range(3)] +  ['eid', '21003-0.0', '31-0.0']).set_index('eid')

    temp = temp.apply(custom_apply, axis = 1)
    df = temp[~temp.isna().any(axis = 1)]
    df.columns = ['3064-0.0', '3062-0.0', '3063-0.0'] +  ['21003-0.0', '31-0.0']

    features_index = df.columns
    features = []
    for elem in features_index:
        if elem != '21003-0.0' and elem != '31-0.0':
            features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
        else:
            features.append(feature_id_to_name[int(elem.split('-')[0])])
    df.columns =  features
    return df
