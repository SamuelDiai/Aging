from .base_processing import path_data, path_dictionary

"""

20112	Illnesses of adopted father
20113	Illnesses of adopted mother
20114	Illnesses of adopted siblings
20107	Illnesses of father
20110	Illnesses of mother
20111	Illnesses of siblings
#1797	Father still alive
#3912	Adopted father still alive
#2946	Father's age
#1807	Father's age at death
#1835	Mother still alive
#3942	Adopted mother still alive
#1845	Mother's age
#3526	Mother's age at death
1873	Number of full brothers
3972	Number of adopted brothers
1883	Number of full sisters
3982	Number of adopted sisters
5057	Number of older siblings
4501	Non-accidental death in close genetic family


"""

def read_family_history_data(instance = 0, **kwargs):


    dict_onehot = {'20107' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease'},
                   '20110' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease'},
                   '20111' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease'},
                   '20112' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease'},
                   '20113' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease'},
                   '20114' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease'}}

    cols_numb_onehot = {'20107' : 10,
                        '20110' : 11,
                        '20111' : 12,
                        '20114' : 7,
                        '20112' : 7,
                        '20113' : 6}


    cols_onehot = [ key + '-%s.%s' % (instance, int_) for key in cols_numb_onehot.keys() for int_ in range(cols_numb_onehot[key])]
    cols_ordinal = []
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

    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continous, **kwargs).set_index('eid')
    temp = temp.dropna(how = 'all')
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
