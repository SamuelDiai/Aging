from .base_processing import path_data, path_dictionary


def read_alcohol_data(instance = 0, **kwargs):

    cols_categorical = [2664, 3859]
    cols = [20117,1558,4407,4418,4429,4440,4451,4462,1568,1578,1588,1598,1608,5364,1618]
    a = pd.read_csv(path_data, usecols = ['eid'] + [str(elem) + '-%s.0' % instance for elem in cols + cols_categorical]).set_index('eid')
    a = a.replace(-6, 0.5)
    a = a[~(a < 0).any(axis = 1)]
    a = a.dropna(how = 'all')
    for column in ['2664-%s.0' % instance] + ['3859-%s.0' % instance]:
        a[column] = a[column].astype('Int64')
        a[column] = a[column].replace(np.nan, 0)

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = a.columns
    features = []
    for elem in features_index:
        features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
    a.columns = features

    cols_categorical = [feature_id_to_name[elem] for elem in cols_categorical]

    for cate in cols_categorical:
        col_ = a[cate + '.0']
        d = pd.get_dummies(col_)
        d.columns = [cate + '.' + str(int(elem)) for elem in d.columns]
        #display(d)
        a = a.drop(columns = [cate + '.0'])
        #display(temp)
        a = a.join(d, how = 'inner')


    a = a.replace(np.nan, 0)
    return a
