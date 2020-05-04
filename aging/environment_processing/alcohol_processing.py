from .base_processing import path_data, path_dictionary


def read_alcohol_data(instances = [0,1,2,3], **kwargs):
    list_df = []
    dict_onehot = {'Reason for reducing amount of alcohol drunk'  : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
                                                                     5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   'Reason former drinker stopped drinking alcohol' : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
                                                                     5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'}}
    cols_categorical = [2664, 3859]
    cols = [20117,1558,4407,4418,4429,4440,4451,4462,1568,1578,1588,1598,1608,5364,1618]

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']

    names_categorical = [feature_id_to_name[elem] for elem in cols_categorical]
    non_selected_cols = [-1, -3]
    for instance in instances :

        a = pd.read_csv(path_data, usecols = ['eid'] + [str(elem) + '-%s.0' % instance for elem in cols + cols_categorical], **kwargs).set_index('eid')
        a = a.replace(-6, 0.5)
        a = a.dropna(how = 'all')
        for column in ['2664-%s.0' % instance] + ['3859-%s.0' % instance]:
            a[column] = a[column].astype('Int32')
            a[column] = a[column].replace(np.nan, 0)


        features_index = a.columns
        features = []
        for elem in features_index:
            features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
        a.columns = features


        for cate in names_categorical:
            col_ = a[cate + '.0']


            d = pd.get_dummies(col_)
            d.columns = [cate + '.' + dict_onehot[cate][int(elem)] for elem in d.columns]
            a = a.drop(columns = [cate + '.0'])
            a = a.join(d, how = 'inner')


        a = a.replace(np.nan, 0)
        a['eid'] = a.index
        a.index = (a.index.astype('str') + '_' + str(instance)).rename('id')
        #list_df.append(a)




        a = a.append(a.reindex(a.columns, axis = 1, fill_value=0))
    #replace -1 and -3 from continuous variables by Nans
    a = a.replace(-1, np.nan)
    a = a.replace(-3, np.nan)
    return a
