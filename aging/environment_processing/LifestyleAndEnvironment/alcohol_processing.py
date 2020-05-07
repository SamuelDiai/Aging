from .base_processing import read_complex_data #, path_data, path_dictionary

    """
    20117	Alcohol drinker status
    1558	Alcohol intake frequency.

    4407	Average monthly red wine intake
    4418	Average monthly champagne plus white wine intake
    4429	Average monthly beer plus cider intake
    4440	Average monthly spirits intake
    4451	Average monthly fortified wine intake
    4462	Average monthly intake of other alcoholic drinks
    1568	Average weekly red wine intake
    1578	Average weekly champagne plus white wine intake
    1588	Average weekly beer plus cider intake
    1598	Average weekly spirits intake
    1608	Average weekly fortified wine intake
    5364	Average weekly intake of other alcoholic drinks

    1618	Alcohol usually taken with meals
    1628	Alcohol intake versus 10 years previously

    2664	Reason for reducing amount of alcohol drunk
    3859	Reason former drinker stopped drinking alcohol

    """

def read_alcohol_data(instances = [0, 1, 2, 3], **kwargs):

    dict_onehot = {'2664'  : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
                                                                     5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '3859' : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
                                                                     5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'}}
    cols_numb_onehot = {'2664' : 1, '3859' : 1}
    cols_ordinal = ['20117', '1558'] + ['1618']
    cols_continuous = ['4407', '4418', '4429', '4440', '4451', '4462', '1568', '1578', '1588', '1598', '1608', '5364']
    cont_fill_na = ['1618'] + cols_continuous
    cols_half_binary = []

    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)

    ## RE-ENCODE :
    df['Alcohol usually taken with meals.0'] = df['Alcohol usually taken with meals.0'].replace(-6, 0.5)
    return df


# def read_alcohol_data(instances = [0,1,2,3], **kwargs):
#
#     dict_onehot = {'Reason for reducing amount of alcohol drunk'  : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
#                                                                      5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'},
#                    'Reason former drinker stopped drinking alcohol' : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
#                                                                      5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'}}
#     cols_categorical = [2664, 3859]
#     cols = [20117,1558,4407,4418,4429,4440,4451,4462,1568,1578,1588,1598,1608,5364,1618]
#
#     df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
#     df_features.set_index('FieldID', inplace = True)
#     feature_id_to_name = df_features.to_dict()['Field']
#
#     names_categorical = [feature_id_to_name[elem] for elem in cols_categorical]
#     non_selected_cols = [-1, -3]
#     for idx, instance in enumerate(instances) :
#
#         a = pd.read_csv(path_data, usecols = ['eid'] + [str(elem) + '-%s.0' % instance for elem in cols + cols_categorical], **kwargs).set_index('eid')
#         a = a.replace(-6, 0.5)
#         a = a.dropna(how = 'all')
#         for column in ['2664-%s.0' % instance] + ['3859-%s.0' % instance]:
#             a[column] = a[column].astype('Int32')
#             a[column] = a[column].replace(np.nan, 0)
#
#
#         features_index = a.columns
#         features = []
#         for elem in features_index:
#             features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
#         a.columns = features
#
#
#         for cate in names_categorical:
#             col_ = a[cate + '.0']
#
#
#             d = pd.get_dummies(col_)
#             d.columns = [cate + '.' + dict_onehot[cate][int(elem)] for elem in d.columns]
#             a = a.drop(columns = [cate + '.0'])
#             a = a.join(d, how = 'inner')
#
#         a = a.replace(np.nan, 0)
#         a['eid'] = a.index
#         a.index = (a.index.astype('str') + '_' + str(instance)).rename('id')
#
#
#         if idx == 0 :
#             df = a
#         else :
#             df = df.append(a.reindex(df.columns, axis = 1, fill_value=0))
#         display(df)
#     #replace -1 and -3 from continuous variables by Nans
#     df = df.replace(-1, np.nan)
#     df = df.replace(-3, np.nan)
#     return df
