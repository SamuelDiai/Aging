from ..base_processing import read_complex_data
"""
1110	Length of mobile phone use
1120	Weekly usage of mobile phone in last 3 months
1130	Hands-free device/speakerphone use with mobile phone in last 3 month
1140	Difference in mobile phone use compared to two years previously
1150	Usual side of head for mobile phone use
2237	Plays computer games
"""


def read_electronic_devices_data(instances = [0, 1, 2, 3], **kwargs):

    cont_fill_na = ['1120', '1130', '1140']
    cols_ordinal = ['1110']
    cols_continuous = ['1120', '1130', '1140', '2237']

    dict_onehot = {'1150' : {1 : 'Left', 2 :'Right', 3: 'Equally left and right', -3 : 'Prefer not to answer', -1 : 'Do not know'}}
    cols_numb_onehot = {'1150' : 1}


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)
    return df


# def read_electronic_device_data(instance = 0, **kwargs):
#
#     cols_onehot = ['1150']
#     cols_ordinal = ['1110']
#     cols_continous = ['1120', '1130', '1140', '2237']
#
#     dict_onehot = {'1150' : {1 : 'Left', 2 :'Right', 3: 'Equally left and right'}}
#
#     """
#         all cols must be strings or int
#         cols_onehot : cols that need one hot encoding
#         cols_ordinal : cols that need to be converted as ints
#         cols_continuous : cols that don't need to be converted as ints
#     """
#
#     ## Format cols :
#     for idx ,elem in enumerate(cols_onehot):
#         if isinstance(elem,(str)):
#             cols_onehot[idx] = elem + '-%s.0' % instance
#         elif isinstance(elem, (int)):
#             cols_onehot[idx] = str(elem) + '-%s.0' % instance
#
#     for idx ,elem in enumerate(cols_ordinal):
#         if isinstance(elem,(str)):
#             cols_ordinal[idx] = elem + '-%s.0' % instance
#         elif isinstance(elem, (int)):
#             cols_ordinal[idx] = str(elem) + '-%s.0' % instance
#
#     for idx ,elem in enumerate(cols_continous):
#         if isinstance(elem,(str)):
#             cols_continous[idx] = elem + '-%s.0' % instance
#         elif isinstance(elem, (int)):
#             cols_continous[idx] = str(elem) + '-%s.0' % instance
#
#     temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continous, **kwargs).set_index('eid')
#     temp = temp.dropna(how = 'all')
#     for column in cols_onehot + cols_ordinal:
#         temp[column] = temp[column].astype('Int64')
#
#     print(cols_onehot)
#     for cate in cols_onehot:
#
#         col_ = temp[cate]
#         d = pd.get_dummies(col_)
#         d = d.drop(columns = [ elem for elem in d.columns if int(elem) < 0 ])
#
#         d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns if int(elem) >= 0]
#         print(d.columns)
#         temp = temp.drop(columns = [cate[:-2] + '.0'])
#         temp = temp.join(d, how = 'inner')
#
#
#     temp['1120-%s.0' % instance] = temp['1130-%s.0' % instance].replace(np.nan, 0)
#     temp['1130-%s.0' % instance] = temp['1130-%s.0' % instance].replace(np.nan, 0)
#     temp['1140-%s.0' % instance] = temp['1130-%s.0' % instance].replace(np.nan, 0)
#
#
#     temp = temp[(temp >= 0).all(axis = 1) & (temp['1140-%s.0' % instance] != 3)]
#     df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
#     df_features.set_index('FieldID', inplace = True)
#     feature_id_to_name = df_features.to_dict()['Field']
#
#
#     features_index = temp.columns
#     features = []
#     for elem in features_index:
#         features.append(feature_id_to_name[int(elem.split('-')[0])] + '.' + elem.split('-')[1][2:])
#     temp.columns = features
#
#     return temp
