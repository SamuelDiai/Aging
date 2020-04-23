from .base_processing import path_data, path_dictionary



"""
1920	Mood swings
1930	Miserableness
1940	Irritability
1950	Sensitivity / hurt feelings
1960	Fed-up feelings
1970	Nervous feelings
1980	Worrier / anxious feelings
1990	Tense / 'highly strung'
2000	Worry too long after embarrassment
2010	Suffer from 'nerves'
2020	Loneliness, isolation
2030	Guilty feelings
2040	Risk taking
4526	Happiness
4537	Work/job satisfaction +> ? big loss due to retired people
4548	Health satisfaction
4559	Family relationship satisfaction
4570	Friendships satisfaction
4581	Financial situation satisfaction
2050	Frequency of depressed mood in last 2 weeks
2060	Frequency of unenthusiasm / disinterest in last 2 weeks
2070	Frequency of tenseness / restlessness in last 2 weeks
2080	Frequency of tiredness / lethargy in last 2 weeks
2090	Seen doctor (GP) for nerves, anxiety, tension or depression
2100	Seen a psychiatrist for nerves, anxiety, tension or depression
4598	Ever depressed for a whole week
4609	Longest period of depression => Put 0 for nans
4620	Number of depression episodes => put 0 for nans
4631	Ever unenthusiastic/disinterested for a whole week
5375	Longest period of unenthusiasm / disinterest => put 0 for nans
5386	Number of unenthusiastic/disinterested episodes => put 0 for nans
4642	Ever manic/hyper for 2 days
4653	Ever highly irritable/argumentative for 2 days
6156	Manic/hyper symptoms
5663	Length of longest manic/irritable episode => put 0 for nans
5674	Severity of manic/irritable episodes
6145	Illness, injury, bereavement, stress in last 2 years

Missing : '4526', '4548', '4559', '4570', '4581' all instances
"""


def read_social_support_data(instance = 0, **kwargs):

    dict_onehot = {'6156' : {11 : 'I was more active than usual', 12 : 'I was more talkative than usual', 13 :'I needed less sleep than usual', 14 : 'I was more creative or had more ideas than usual',
                             15 : 'All of the above', -7 : 'None of the above', 0 : 'No symptoms'},
                   '6145' : {1 : 'Serious illness, injury or assault to yourself', 2 : 'Serious illness, injury or assault of a close relative', 3 : 'Death of a close relative', 4 : 'Death of a spouse or partner',
                             5 : 'Marital separation/divorce', 6 : 'Financial difficulties', -7 : 'None of the above'}}

    cols_onehot_6145 = ['6145-%s.0' % instance,'6145-%s.1' % instance, '6145-%s.2' % instance,'6145-%s.3' % instance, '6145-%s.4' % instance, '6145-%s.5' % instance]
    cols_onehot_6156 = ['6156-%s.0' % instance,'6156-%s.1' % instance,'6156-%s.2' % instance,'6156-%s.3' % instance ]
    cols_ordinal = ['1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020', '2030', '2040',
                    #'4526', '4548', '4559', '4570', '4581',
                    '2050', '2060', '2070', '2080', '2090', '2100',
                    '4598', '4609', '4620', '4631', '5375', '5386', '4642', '4653', '5663']
    cols_continous = []

    """
        all cols must be strings or int
        cols_onehot : cols that need one hot encoding
        cols_ordinal : cols that need to be converted as ints
        cols_continuous : cols that don't need to be converted as ints
    """


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

    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot_6145 + cols_onehot_6156 + cols_ordinal + cols_continous, **kwargs).set_index('eid')
    for column in cols_onehot_6156 + cols_onehot_6145 + cols_ordinal:
        temp[column] = temp[column].astype('Int64')


    for idx, cate in enumerate(cols_onehot_6145):
        col_ = temp[cate]
        if idx == 0 :
            d = pd.get_dummies(col_)
            d = d.drop(columns = [ elem for elem in d.columns if int(elem) == -3 ])
            d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns if int(elem) != -3]
        else :
            d_ = pd.get_dummies(col_)
            d_ = d_.drop(columns = [ elem for elem in d_.columns if int(elem) == -3 ])
            d_.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d_.columns if int(elem) != -3]
            d[d_.columns] = d[d_.columns].add(d_)
        temp = temp.drop(columns = [cate[:-2] + '.%s' % idx])
    temp = temp.join(d, how = 'inner')

    for idx, cate in enumerate(cols_onehot_6156):
        col_ = temp[cate]
        if idx == 0 :
            d = pd.get_dummies(col_)
            d = d.drop(columns = [ elem for elem in d.columns if int(elem) == -3 ])
            d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns if int(elem) != -3]
        else :
            d_ = pd.get_dummies(col_)
            d_ = d_.drop(columns = [ elem for elem in d_.columns if int(elem) == -3 ])
            d_.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d_.columns if int(elem) != -3]
            d[d_.columns] = d[d_.columns].add(d_)
        temp = temp.drop(columns = [cate[:-2] + '.%s' % idx])
    temp = temp.join(d, how = 'inner')

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = temp.columns
    features = []
    for elem in features_index:
        split = elem.split('-%s' % instance)
        features.append(feature_id_to_name[int(split[0])] + split[1])
    temp.columns = features

    temp['Longest period of depression.0'] = temp['Longest period of depression.0'].replace(np.nan, 0)
    temp['Number of depression episodes.0'] = temp['Number of depression episodes.0'].replace(np.nan, 0)
    temp['Number of unenthusiastic/disinterested episodes.0'] = temp['Number of unenthusiastic/disinterested episodes.0'].replace(np.nan, 0)
    temp['Longest period of unenthusiasm / disinterest.0'] = temp['Longest period of unenthusiasm / disinterest.0'].replace(np.nan, 0)
    temp['Longest period of depression.0'] = temp['Longest period of depression.0'].replace(np.nan, 0)
    temp['Length of longest manic/irritable episode.0'] = temp['Length of longest manic/irritable episode.0'].replace(np.nan, 0)

    temp = temp[(temp < 0).any(axis = 1)]
    temp = temp.dropna(how = 'any')

    return temp, d
