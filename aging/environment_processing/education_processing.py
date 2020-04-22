from .base_processing import path_data, path_dictionary

"""
6138 Qualifications
845	Age completed full time education
"""


def read_education_data(instance = 0, **kwargs):

    dict_onehot  = {'6138' : {1 : 'College or University degree', 2 : 'A levels/AS levels or equivalent', 3 : 'O levels/GCSEs or equivalent',
                              4 : 'CSEs or equivalent', 5 : 'NVQ or HND or HNC or equivalent', 6 : 'Other professional qualifications eg: nursing, teaching',
                             -7 : 'None of the above'}}

    cols_onehot = ['6138']
    cols_ordinal = ['845']
    cols_continous = []
    """
        all cols must be strings or int
        cols_onehot : cols that need one hot encoding
        cols_ordinal : cols that need to be converted as ints
        cols_continuous : cols that don't need to be converted as ints
    """

    ## Format cols :
    for idx ,elem in enumerate(cols_onehot):
        if isinstance(elem,(str)):
            cols_onehot[idx] = elem + '-%s.0' % instance
        elif isinstance(elem, (int)):
            cols_onehot[idx] = str(elem) + '-%s.0' % instance

    for idx ,elem in enumerate(cols_ordinal):
        if isinstance(elem,(str)):
            cols_ordinal[idx] = elem + '-%s.0' % instance
        elif isinstance(elem, (int)):
            cols_ordinal[idx] = str(elem) + '-%s.0' % instance

    for idx ,elem in enumerate(cols_continous):
        if isinstance(elem,(str)):
            cols_continous[idx] = elem + '-%s.0' % instance
        elif isinstance(elem, (int)):
            cols_continous[idx] = str(elem) + '-%s.0' % instance

    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continous, **kwargs).set_index('eid')

    temp = temp[~(temp < 0).any(axis = 1)]
    temp = temp.dropna(how = 'any')


    for column in cols_onehot + cols_ordinal:
        temp[column] = temp[column].astype('Int64')

    for cate in cols_onehot:
        col_ = temp[cate]
        d = pd.get_dummies(col_)
        try :
            d = d.drop(columns = [-3])
        except KeyError:
            d = d

        d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns]
        temp = temp.drop(columns = [cate[:-2] + '.0'])
        temp = temp.join(d, how = 'inner')

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = temp.columns
    features = []
    for elem in features_index:
        split = elem.split('-%s' % instance)
        features.append(feature_id_to_name[int(split[0])] + split[1])
    temp.columns = features


    return temp
