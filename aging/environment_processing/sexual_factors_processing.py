from .base_processing import path_data, path_dictionary
"""
2139	Age first had sexual intercourse
2149	Lifetime number of sexual partners
2159	Ever had same-sex intercourse
3669	Lifetime number of same-sex sexual partners
"""

def read_sexual_factors_data(instance = 0, **kwargs):

    cols_onehot = []
    cols_ordinal = ['2129', '2139', '2149', '2159', '3669']
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
    temp = temp[temp['2129-%s.0' % instance] == 1]
    temp = temp.drop(columns = ['2129-%s.0' % instance])

    for column in cols_onehot + cols_ordinal:
        if column != '2129-%s.0' % instance:
            temp[column] = temp[column].astype('Int64')

    temp['3669-%s.0' % instance] = temp['3669-%s.0' % instance].replace(np.nan, 0)
    temp['2149-%s.0' % instance] = temp['2149-%s.0' % instance].replace(np.nan, 0)
    temp['2159-%s.0' % instance] = temp['2159-%s.0' % instance].replace(np.nan, 0)

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = temp.columns
    features = []
    for elem in features_index:
        split = elem.split('-%s' % instance)
        features.append(feature_id_to_name[int(split[0])] + split[1])
    temp.columns = features

    temp['Ever had sexual intercourse.0'] = temp['Age first had sexual intercourse.0']
    temp['Ever had sexual intercourse.0'][temp['Ever had sexual intercourse.0'] > 0] = 1
    temp['Ever had sexual intercourse.0'][temp['Ever had sexual intercourse.0'] == -2] = 0

    temp = temp.drop(columns = ['Age first had sexual intercourse.0'])
    temp = temp[~(temp < 0).any(axis = 1)]

    return temp
