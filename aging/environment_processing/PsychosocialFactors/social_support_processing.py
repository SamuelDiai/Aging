from ..base_processing import read_complex_data

"""
1031	Frequency of friend/family visits
6160	Leisure/social activities
2110	Able to confide

"""

def read_social_support_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {'6160' : {1 : 'Sports club or gym',  2 : 'Pub or social club', 3 : 'Religious group', 4 : 'Adult education class', 5 : 'Other group activity',
                            -7 : 'None of the above', -3 :'Prefer not to answer'}}

    cols_ordinal = ['2110']
    cols_numb_onehot = {'6160' : 5}
    cols_continuous = ['1031']
    cont_fill_na = []
    cols_half_binary = []


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df

# def read_social_support_data(instance = 0, **kwargs):
#
#     dict_onehot = {'6160' : {1 : 'Sports club or gym',  2 : 'Pub or social club', 3 : 'Religious group', 4 : 'Adult education class', 5 : 'Other group activity',
#                             -7 : 'None of the above'}}
#
#     cols_onehot = ['6160-%s.0' %instance, '6160-%s.1' %instance, '6160-%s.2' %instance, '6160-%s.3' %instance, '6160-%s.4' %instance]
#     cols_ordinal = ['1031', '2110']
#     cols_continous = []
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
#     ## Format cols :
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
#     for idx, cate in enumerate(cols_onehot):
#         col_ = temp[cate]
#         d = pd.get_dummies(col_)
#         d = d.drop(columns = [ elem for elem in d.columns if int(elem) == -3 ])
#         d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns if int(elem) != -3]
#         temp = temp.drop(columns = [cate[:-2] + '.%s' % idx])
#
#         if idx == 0:
#             d_ = d
#         else :
#             d_[d.columns] = d_[d.columns].add(d)
#     #display(d_)
#     temp = temp.join(d_, how = 'inner')
#     #print(temp.columns)
#
#
#     temp = temp[temp['1031-%s.0' % instance] != 7]
#     temp = temp[~(temp < 0).any(axis = 1)]
#     temp = temp.dropna(how = 'any')
#
#
#     df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
#     df_features.set_index('FieldID', inplace = True)
#     feature_id_to_name = df_features.to_dict()['Field']
#
#
#     features_index = temp.columns
#     features = []
#     for elem in features_index:
#         print(elem)
#         split = elem.split('-%s' % instance)
#         print(split)
#         features.append(feature_id_to_name[int(split[0])] + split[1])
#     temp.columns = features
#
#
#     return temp
