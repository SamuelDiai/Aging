"""
EYE :

2207	Wears glasses or contact lenses
2217	Age started wearing glasses or contact lenses
6147	Reason for glasses/contact lenses => 6
5843	Which eye(s) affected by myopia (short sight)
5832	Which eye(s) affected by hypermetropia (long sight)
5610	Which eye(s) affected by presbyopia
5855	Which eye(s) affected by astigmatism
6205	Which eye(s) affected by strabismus (squint)
5408	Which eye(s) affected by amblyopia (lazy eye)
5877	Which eye(s) affected by other eye condition
5934	Which eye(s) affected by other serious eye condition
2227	Other eye problems
6148	Eye problems/disorders => 5
5890	Which eye(s) affected by diabetes-related eye disease
6119	Which eye(s) affected by glaucoma
5419	Which eye(s) affected by injury or trauma resulting in loss of vision
5441	Which eye(s) are affected by cataract
5912	Which eye(s) affected by macular degeneration
5901	Age when diabetes-related eye disease diagnosed
4689	Age glaucoma diagnosed
5430	Age when loss of vision due to injury or trauma diagnosed
4700	Age cataract diagnosed
5923	Age macular degeneration diagnosed
5945	Age other serious eye condition diagnosed

"""

def read_eye_history_data(instance = 0, **kwargs):

    dict_onehot = {'6148' : {1: 'Diabetes related eye disease',
                             2: 'Glaucoma',
                             3: 'Injury or trauma resulting in loss of vision',
                             4: 'Cataract',
                             5: 'Macular degeneration',
                             6: 'Other serious eye condition',
                             -7: 'None of the above'},
                   '6147' : {1: 'For short-sightedness, i.e. only or mainly for distance viewing such as driving, cinema etc (called myopia)',
                             2: 'For long-sightedness, i.e. for distance and near, but particularly for near tasks like reading (called hypermetropia)',
                             3: 'For just reading/near work as you are getting older (called presbyopia)',
                             4: 'For astigmatism',
                             5: 'For a squint or turn in an eye since childhood (called strabismus)',
                             6: 'For a lazy eye or an eye with poor vision since childhood (called amblyopia)',
                             7: 'Other eye condition'},
                   '5843' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5832' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5610' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5855' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '6205' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5408' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5877' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5934' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5890' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '6119' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5419' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5441' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                   '5912' : {1 :'Right eye',
                             2 :'Left eye',
                             3 :'Both eyes'},
                }

    cols_numb_onehot = {'6147' : 6,
                        '6148' : 5,
                        '5843' : 1,
                        '5832' : 1,
                        '5610' : 1,
                        '5855' : 1,
                        '6205' : 1,
                        '5408' : 1,
                        '5877' : 1,
                        '5934' : 1,
                        '5890' : 1,
                        '6119' : 1,
                        '5419' : 1,
                        '5441' : 1,
                        '5912' : 1}


    cols_onehot = [ key + '-%s.%s' % (instance, int_) for key in cols_numb_onehot.keys() for int_ in range(cols_numb_onehot[key])]
    cols_ordinal = ['2207', '2217','2227', '5901', '4689', '5430', '4700', '5923', '5945']
    cols_continous = []

    """
        all cols must be strings or int
        cols_onehot : cols that need one hot encoding
        cols_ordinal : cols that need to be converted as ints
        cols_continuous : cols that don't need to be converted as ints
    """


    ## Format cols :

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

    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continous, **kwargs)

    temp = temp.set_index('eid')
    temp = temp.dropna(how = 'all')

    display(temp)
    for column in cols_onehot + cols_ordinal:
        temp[column] = temp[column].astype('Int64')

    for col in cols_numb_onehot.keys():

        for idx in range(cols_numb_onehot[col]):
            cate = col + '-%s.%s' % (instance, idx)
            d = pd.get_dummies(temp[cate])
            d = d.drop(columns = [ elem for elem in d.columns if int(elem) < 0 ])
            d.columns = [col + '-%s'%instance + '.' + dict_onehot[col][int(elem)] for elem in d.columns ]
            temp = temp.drop(columns = [cate])

            if idx == 0:
                d_ = d
            else :
                common_cols = d.columns.intersection(d_.columns)
                remaining_cols = d.columns.difference(common_cols)
                if len(common_cols) > 0 :
                    d_[common_cols] = d_[common_cols].add(d[common_cols])
                for col_ in remaining_cols:
                    d_[col_] = d[col_]
        temp = temp.join(d_, how = 'inner')


    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = temp.columns
    features = []
    for elem in features_index:
        split = elem.split('-%s' % instance)
        features.append(feature_id_to_name[int(split[0])] + split[1])
    temp.columns = features


    return temp
