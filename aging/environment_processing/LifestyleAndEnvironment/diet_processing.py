from .base_processing import read_complex_data

    """
    1289	Cooked vegetable intake
    1299	Salad / raw vegetable intake
    1309	Fresh fruit intake
    1319	Dried fruit intake
    1329	Oily fish intake
    1339	Non-oily fish intake
    1349	Processed meat intake
    1359	Poultry intake
    1369	Beef intake
    1379	Lamb/mutton intake
    1389	Pork intake
    1408	Cheese intake
    1438	Bread intake
    1458	Cereal intake
    1478	Salt added to food
    1488	Tea intake
    1498	Coffee intake => MISSING !!
    1518	Hot drink temperature
    1528	Water intake
    1548	Variation in diet


    6144	Never eat eggs, dairy, wheat, sugar
    1418	Milk type used
    1428	Spread type
    2654	Non-butter spread type details
    1448	Bread type
    1468	Cereal type
    1508	Coffee type
    1538	Major dietary changes in the last 5 years


    """

def read_diet_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'1418' : {1: 'Full cream', 2 : 'Semi-skimmed', 3 : 'Skimmed', 4 : 'Soya', 5 : 'Other type of milk', 6 : 'Never/rarely have milk', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1428'  : {1 :'Butter/spreadable butter', 3 : 'Other type of spread/margarine', 0 :'Never/rarely use spread', 2 : 'Flora Pro-Active/Benecol', -1 : 'Do not know',-3 : 'Prefer not to answer'},
                   '1448' : {1 : 'White', 2 : 'Brown', 3 : 'Wholemeal or wholegrain', 4 : 'Other type of bread', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1468' : {1 : 'Bran cereal (e.g. All Bran, Branflakes)', 2 : 'Biscuit cereal (e.g. Weetabix)', 3 : 'Oat cereal (e.g. Ready Brek, porridge)', 4 : 'Muesli', 5 : 'Other (e.g. Cornflakes, Frosties)',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1508' : {1 : 'Decaffeinated coffee (any type)', 2 : 'Instant coffee', 3 : 'Ground coffee (include espresso, filter etc)', 4 : 'Other type of coffee', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '2654' : {4 : 'Soft (tub) margarine',5 : 'Hard (block) margarine',  6 : 'Olive oil based spread (eg: Bertolli)', 7 : 'Polyunsaturated/sunflower oil based spread (eg: Flora)',
                         2 : 'Flora Pro-Active or Benecol', 8 : 'Other low or reduced fat spread', 9 : 'Other type of spread/margarine', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6144' : {1 : 'Eggs or foods containing eggs', 2 : 'Dairy products', 3 : 'Wheat products', 4 : 'Sugar or foods/drinks containing sugar', 5 : 'I eat all of the above', -3 : 'Prefer not to answer'},
                   '1508' : {1 : 'Decaffeinated coffee (any type)', 2 : 'Instant coffee', 3 : 'Ground coffee (include espresso, filter etc)', 4 : 'Other type of coffee', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1538' : {0 : 'No', 1 : 'Yes, because of illness', 2 : 'Yes, because of other reasons', -3 : 'Prefer not to answer'}
                   }

    cols_numb_onehot = {'1418' : 1,
                        '1428' : 1,
                        '1448' : 1,
                        '1468' : 1,
                        '2654' : 1,
                        '6144' : 4,
                        '1508' : 1,
                        '1538' : 1}

    cols_continuous = ['1289','1299','1309','1319','1329','1339','1349','1359','1369','1379','1389','1408','1438','1458','1478','1488', #'1498',
                           '1518','1528','1548']
    cols_ordinal = []
    cont_fill_na = []



    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)
    df = df.replace(-10, 0)
    return df

# def read_diet_data(instances = [0, 1, 2, 3], **kwargs):
#     df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
#     df_features.set_index('FieldID', inplace = True)
#     feature_id_to_name = df_features.to_dict()['Field']
#
#     dict_onehot = {'1418' : {1: 'Full cream', 2 : 'Semi-skimmed', 3 : 'Skimmed', 4 : 'Soya', 5 : 'Other type of milk', 6 : 'Never/rarely have milk', -1 : 'Do not know'},
#                    '1428'  : {1 :'Butter/spreadable butter', 3 : 'Other type of spread/margarine', 0 :'Never/rarely use spread', 2 : 'Flora Pro-Active/Benecol', -1 : 'Do not know'},
#                    '1448' : {1 : 'White', 2 : 'Brown', 3 : 'Wholemeal or wholegrain', 4 : 'Other type of bread', -1 : 'Do not know', -3 : 'Prefer not to answer'},
#                    '1468' : {1 : 'Bran cereal (e.g. All Bran, Branflakes)', 2 : 'Biscuit cereal (e.g. Weetabix)', 3 : 'Oat cereal (e.g. Ready Brek, porridge)', 4 : 'Muesli', 5 : 'Other (e.g. Cornflakes, Frosties)',
#                              -1 : 'Do not know', -3 : 'Prefer not to answer'},
#                    '1508' : {1 : 'Decaffeinated coffee (any type)', 2 : 'Instant coffee', 3 : 'Ground coffee (include espresso, filter etc)', 4 : 'Other type of coffee', -1 : 'Do not know', -3 : 'Prefer not to answer'},
#                    '2654' : {4 : 'Soft (tub) margarine',5 : 'Hard (block) margarine',  6 : 'Olive oil based spread (eg: Bertolli)', 7 : 'Polyunsaturated/sunflower oil based spread (eg: Flora)',
#                          2 : 'Flora Pro-Active or Benecol', 8 : 'Other low or reduced fat spread', 9 : 'Other type of spread/margarine', -1 : 'Do not know', -3 : 'Prefer not to answer'},
#                    '6144' : {1 : 'Eggs or foods containing eggs', 2 : 'Dairy products', 3 : 'Wheat products', 4 : 'Sugar or foods/drinks containing sugar', 5 : 'I eat all of the above', -3 : 'Prefer not to answer'},
#                    '1508' : {1 : 'Decaffeinated coffee (any type)', 2 : 'Instant coffee', 3 : 'Ground coffee (include espresso, filter etc)', 4 : 'Other type of coffee', -1 : 'Do not know', -3 : 'Prefer not to answer'},
#                    '1538' : {0 : 'No', 1 : 'Yes, because of illness', 2 : 'Yes, because of other reasons', -3 : 'Prefer not to answer'}
#                    }
#
#     cols_numb_onehot = {'1418' : 1,
#                         '1428' : 1,
#                         '1448' : 1,
#                         '1468' : 1,
#                         '2654' : 1,
#                         '6144' : 4,
#                         '1508' : 1,
#                         '1538' : 1}
#
#
#     """
#         all cols must be strings or int
#         cols_onehot : cols that need one hot encoding
#         cols_ordinal : cols that need to be converted as ints
#         cols_continuous : cols that don't need to be converted as ints
#     """
#
#
#
#
#     for idx_instance, instance in enumerate(instances) :
#
#
#         cols_onehot = [ key + '-%s.%s' % (instance, int_) for key in cols_numb_onehot.keys() for int_ in range(cols_numb_onehot[key])]
#         cols_ordinal = []
#         cols_continuous = ['1289','1299','1309','1319','1329','1339','1349','1359','1369','1379','1389','1408','1438','1458','1478','1488', #'1498',
#                            '1518','1528','1548']
#
#         for idx ,elem in enumerate(cols_ordinal):
#             if isinstance(elem,(str)):
#                 cols_ordinal[idx] = elem + '-%s.0' % instance
#             elif isinstance(elem, (int)):
#                 cols_ordinal[idx] = str(elem) + '-%s.0' % instance
#
#         for idx ,elem in enumerate(cols_continuous):
#             if isinstance(elem,(str)):
#                 cols_continuous[idx] = elem + '-%s.0' % instance
#             elif isinstance(elem, (int)):
#                 cols_continuous[idx] = str(elem) + '-%s.0' % instance
#
#         temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continuous, **kwargs).set_index('eid')
#         temp = temp.dropna(how = 'all')
#         for column in cols_onehot + cols_ordinal:
#             temp[column] = temp[column].astype('Int32')
#
#
#         temp['1408-%s.0' % instance] = temp['1408-%s.0' % instance].fillna(0)
#
#
#         temp[cols_continuous] = temp[cols_continuous].replace(-10, 0)
#         display(temp[temp.isna().any(axis = 1)])
#
#         for col in cols_numb_onehot.keys():
#
#             for idx in range(cols_numb_onehot[col]):
#                 cate = col + '-%s.%s' % (instance, idx)
#                 d = pd.get_dummies(temp[cate])
#                 d.columns = [col + '-%s'%instance + '.' + dict_onehot[col][int(elem)] for elem in d.columns ]
#                 temp = temp.drop(columns = [cate])
#
#                 if idx == 0:
#                     d_ = d
#                 else :
#                     common_cols = d.columns.intersection(d_.columns)
#                     remaining_cols = d.columns.difference(common_cols)
#                     if len(common_cols) > 0 :
#                         d_[common_cols] = d_[common_cols].add(d[common_cols])
#                     for col_ in remaining_cols:
#                         d_[col_] = d[col_]
#             temp = temp.join(d_, how = 'inner')
#
#
#         features_index = temp.columns
#         features = []
#         for elem in features_index:
#             split = elem.split('-%s' % instance)
#             features.append(feature_id_to_name[int(split[0])] + split[1])
#         temp.columns = features
#
#         temp['eid'] = temp.index
#         temp.index = (temp.index.astype('str') + '_' + str(instance)).rename('id')
#         if idx_instance == 0 :
#             df = temp
#         else :
#             df = df.append(temp.reindex(df.columns, axis = 1, fill_value=0))
#
#     df = df.replace(-1, np.nan)
#     df = df.replace(-3, np.nan)
#     return df
