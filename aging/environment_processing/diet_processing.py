from .base_processing import path_data, path_dictionary




def read_diet_data(instance = 0, **kwargs):


    dict_onehot = {'Milk type used' : {1: 'Full cream', 2	: 'Semi-skimmed', 3 : 'Skimmed', 4 : 'Soya', 5 : 'Other type of milk', 6 : 'Never/rarely have milk'},
                   'Spread type'  : {1 :'Butter/spreadable butter', 3 : 'Other type of spread/margarine', 0 :'Never/rarely use spread', 2 : 'Flora Pro-Active/Benecol'}
                   }


    cols_continuous = ['1289','1299','1309','1319','1329','1339','1349','1359','1369','1379','1389','1408','1438','1458','1478','1488',
                           '1518','1528','1548']
    cols_continuous = [elem + '-%s.0' % instance for elem in cols_continuous]

    cols_categorical= [ '1418','1428']
    cols_categorical = [elem + '-%s.0' % instance for elem in cols_categorical]


    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_continuous + cols_categorical, **kwargs).set_index('eid')
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
    print(cols_categorical)
    for cate in cols_categorical:
        col_ = temp[cate + '.0']
        d = pd.get_dummies(col_)

        d.columns = [cate + '.' + dict_onehot[cate][int(elem)] for elem in d.columns]
        #display(d)
        temp = temp.drop(columns = [cate + '.0'])
        #display(temp)
        temp = temp.join(d, how = 'inner')

    return temp
#match name

#for col_cate in cols_categorical:
#    dim =  temp[col_cate].groupby([col_cate]).count()
