from .base_processing import path_data, path_dictionary




def read_diet_data(**kwargs):

    cols_continuous = ['1289-0.0','1299-0.0','1309-0.0','1319-0.0','1329-0.0','1339-0.0','1349-0.0','1359-0.0','1369-0.0','1379-0.0','1389-0.0','1408-0.0','1438-0.0','1458-0.0','1478-0.0','1488-0.0',
                           '1518-0.0','1528-0.0','1548-0.0']

    cols_categorical= [ '1418-0.0','1428-0.0']


    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_continuous + cols_categorical, nrows = 10000).set_index('eid')
    temp[cols_continuous] = temp[cols_continuous].replace(-10, 0)
    temp = temp[temp >= 0]
    temp = temp.dropna(how = 'any')

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = temp.columns
    features = []
    for elem in features_index:
         features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
    temp.columns = features

    cols_categorical = [feature_id_to_name[int(elem.split('-')[0])] for elem in cols_categorical]

    for cate in cols_categorical:
        col_ = temp[cate + '.0']
        d = pd.get_dummies(col_)
        d.columns = [cate + '.' + str(int(elem)) for elem in d.columns]
        #display(d)
        temp = temp.drop(columns = [cate + '.0'])
        #display(temp)
        temp = temp.join(d, how = 'inner')

    return temp
#match name

#for col_cate in cols_categorical:
#    dim =  temp[col_cate].groupby([col_cate]).count()
