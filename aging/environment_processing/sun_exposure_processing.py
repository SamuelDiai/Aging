from .base_processing import path_data, path_dictionary
"""
1050	Time spend outdoors in summer
1060	Time spent outdoors in winter
1717	Skin colour
1727	Ease of skin tanning
1737	Childhood sunburn occasions
1747	Hair colour (natural, before greying)
1757	Facial ageing
2267	Use of sun/uv protection
2277	Frequency of solarium/sunlamp use
"""


def read_sun_exposure_data(instance = 0, **kwargs):

    cols_onehot = ['1747', '1757']
    cols_ordinal = ['1050', '1060', '1717', '1727', '1737', '2267', '2277']
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
    temp = temp.dropna(how = 'all')
    for column in cols_onehot + cols_ordinal:
        temp[column] = temp[column].astype('Int64')

    for cate in cols_onehot:
        col_ = temp[cate]
        d = pd.get_dummies(col_)
        #d = d.drop(columns = [ elem for elem in d.columns if int(elem) < 0 ])
        d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns
        temp = temp.drop(columns = [cate[:-2] + '.0'])
        temp = temp.join(d, how = 'inner')

    temp = temp.drop(columns = ['1747-%s.-1' % instance, '1747-%s.-3' % instance, '1757-%s.-3' % instance])

    temp['1060-%s.0' % instance] = temp['1060-%s.0' % instance].replace(-10, 0)
    temp['1050-%s.0' % instance] = temp['1050-%s.0' % instance].replace(-10, 0)
    temp = temp[(temp >= 0).all(axis = 1)]

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = temp.columns
    features = []
    for elem in features_index:
        print(elem)
        split = elem.split('-%s' % instance)
        print(split)
        features.append(feature_id_to_name[int(split[0])] + split[1])
    temp.columns = features


    return temp
