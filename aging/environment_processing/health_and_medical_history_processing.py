from .base_processing import read_complex_data

def read_eye_history_data(instances = [0, 1, 2, 3], **kwargs):
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
    dict_onehot = {'6148' : {1: 'Diabetes related eye disease',
                             2: 'Glaucoma',
                             3: 'Injury or trauma resulting in loss of vision',
                             4: 'Cataract',
                             5: 'Macular degeneration',
                             6: 'Other serious eye condition',
                             -1: 'Do not know',
                             -7: 'None of the above',
                             -3: 'Prefer not to answer'},
                   '6147' : {1: 'For short-sightedness, i.e. only or mainly for distance viewing such as driving, cinema etc (called myopia)',
                             2: 'For long-sightedness, i.e. for distance and near, but particularly for near tasks like reading (called hypermetropia)',
                             3: 'For just reading/near work as you are getting older (called presbyopia)',
                             4: 'For astigmatism',
                             5: 'For a squint or turn in an eye since childhood (called strabismus)',
                             6: 'For a lazy eye or an eye with poor vision since childhood (called amblyopia)',
                             7: 'Other eye condition',
                             -1: 'Do not know',
                             -3: 'Prefer not to answer'},
                   '5843' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5832' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5610' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5855' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '6205' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5408' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5877' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5934' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5890' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '6119' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5419' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5441' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5912' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},}

    cols_numb_onehot = {'6147' : 6,'6148' : 5,
                        '5843' : 1,'5832' : 1,'5610' : 1,'5855' : 1,'6205' : 1,'5408' : 1,'5877' : 1,
                        '5934' : 1,'5890' : 1,'6119' : 1,'5419' : 1,'5441' : 1,'5912' : 1}
    cols_ordinal = ['2207', '2217', '2227']
    cols_continuous = ['5901', '4689', '5430', '4700', '5923', '5945']
    cont_fill_na = ['2217', '5901', '4689', '5430', '4700', '5923', '5945']
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

def read_mouth_teeth_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {'6149' : {1 : 'Mouth ulcers', 2 : 'Painful gums', 3 : 'Bleeding gums', 4 : 'Loose teeth', 5 : 'Toothache',6 :'Dentures',-7 : 'None of the above',
                            -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'6149' : 6}
    cols_ordinal = []
    cols_continuous = []
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


def read_general_health_data(instances = [0, 1, 2, 3], **kwargs):
    """ 2178	Overall health rating
        2188	Long-standing illness, disability or infirmity
        2296	Falls in the last year
        2306	Weight change compared with 1 year ago
    """

    dict_onehot = {'2306' : {0 : 'No - weigh about the same', 2 : 'Yes - gained weight', 3 : 'Yes - lost weight',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'2306' : 1}
    cols_ordinal = ['2188', '2296']
    cols_continuous = ['2178']
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

def read_breathing_data(instances = [0, 1, 2, 3], **kwargs):
    """
    2316	Wheeze or whistling in the chest in last year
    4717	Shortness of breath walking on level ground

    """

    dict_onehot = {}

    cols_numb_onehot = {}
    cols_ordinal = ['2316', '4717']
    cols_continuous = []
    cont_fill_na = []
    cols_half_binary = ['2316', '4717']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df

def read_claudication_data(instances = [0, 1, 2, 3], **kwargs):
    """
    4728	Leg pain on walking
    5452	Leg pain when standing still or sitting
    5463	Leg pain in calf/calves
    5474	Leg pain when walking uphill or hurrying
    5485	Leg pain when walking normally
    5496	Leg pain when walking ever disappears while walking
    5507	Leg pain on walking : action taken
    5518	Leg pain on walking : effect of standing still
    5529	Surgery on leg arteries (other than for varicose veins)
    5540	Surgery/amputation of toe or leg

    """

    dict_onehot = {'5518' : {1 : 'Pain usually continues for more than 10 minutes', 2 : 'Pain usually disappears in less than 10 minutes', -1 : 'Do not know', -3 : 'Prefer not to answer'}}
    cols_numb_onehot = {'5518' : 1}
    cols_ordinal = ['4728', '5452', '5463', '5474', '5485', '5496', '5507', '5529', '5540']
    cols_continuous = []
    cont_fill_na = ['5452', '5463', '5474', '5485', '5496', '5507', '5529', '5540']
    cols_half_binary = ['5452', '5463', '5474', '5485', '5496']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df


def read_general_pain_data(instances = [0, 1, 2, 3], **kwargs):
    """
    6159	Pain type(s) experienced in last month
    3799	Headaches for 3+ months
    4067	Facial pains for 3+ months
    3404	Neck/shoulder pain for 3+ months
    3571	Back pain for 3+ months
    3741	Stomach/abdominal pain for 3+ months
    3414	Hip pain for 3+ months
    3773	Knee pain for 3+ months
    2956	General pain for 3+ months
    """

    dict_onehot = {'6159' : {1 :'Headache', 2 : 'Facial pain', 3 : 'Neck or shoulder pain', 4 : 'Back pain',
                             5 : 'Stomach or abdominal pain', 6 : 'Hip pain', 7 :'Knee pain', 8 : 'Pain all over the body',
                             -7 : 'None of the above', -3 : 'Prefer not to answer'},

                  }
    cols_numb_onehot = {'6159' : 7}
    cols_ordinal = ['3799', '4067', '3404', '3571', '3741', '3414', '3773', '2956']
    cols_continuous = []
    cont_fill_na = ['3799', '4067', '3404', '3571', '3741', '3414', '3773', '2956']
    cols_half_binary = ['3799', '4067', '3404', '3571', '3741', '3414', '3773', '2956']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df


def read_chest_pain_data(instances = [0, 1, 2, 3], **kwargs):
    """
    2335	Chest pain or discomfort
    3606	Chest pain or discomfort walking normally
    3616	Chest pain due to walking ceases when standing still
    3751	Chest pain or discomfort when walking uphill or hurrying

    """

    dict_onehot = {}
    cols_numb_onehot = {}
    cols_ordinal = ['2335', '3606', '3616', '3751']
    cols_continuous = []
    cont_fill_na = ['3606', '3616', '3751']
    cols_half_binary = {'2335' : 0.5, '3616' : 0.5, '3606' : 2, '3751' : 2}


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df


def read_cancer_screening_data(instances = [0, 1, 2, 3], **kwargs):
    """
    2345	Ever had bowel cancer screening
    2355	Most recent bowel cancer screening
    2365	Ever had prostate specific antigen (PSA) test
    3809	Time since last prostate specific antigen (PSA) test
    """

    dict_onehot = {}
    cols_numb_onehot = {}
    cols_ordinal = ['2345', '2365']
    cols_continuous = ['2355', '3809']
    cont_fill_na = {'2355' : 100,  '3809' : 100}
    cols_half_binary = []


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    df = df.replace(-10, 0)
    return df

def read_medication_data(instances = [0, 1, 2, 3], **kwargs):
    """
    6177	Medication for cholesterol, blood pressure or diabetes
    6153	Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones
    6154	Medication for pain relief, constipation, heartburn
    6155	Vitamin and mineral supplements
    6179	Mineral and other dietary supplements
    """

    dict_onehot = {'6177' : {1 : 'Cholesterol lowering medication', 2 : 'Blood pressure medication', 3 : 'Insulin',
                             -7 : 'None of the above', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6153' : {1 : 'Cholesterol lowering medication', 2 : 'Blood pressure medication', 3 : 'Insulin',
                             4 : 'Hormone replacement therapy', 5 : 'Oral contraceptive pill or minipill', -7 : 'None of the above',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6154' : {1: 'Aspirin', 2 : 'Ibuprofen (e.g. Nurofen)', 3 : 'Paracetamol', 4 : 'Ranitidine (e.g. Zantac)',
                             5 : 'Omeprazole (e.g. Zanprol)', 6 : 'Laxatives (e.g. Dulcolax, Senokot)', -7 : 'None of the above',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6155' : {1 : 'Vitamin A', 2 : 'Vitamin B', 3 : 'Vitamin C', 4 : 'Vitamin D', 5 : 'Vitamin E', 6 : 'Folic acid or Folate (Vit B9)',
                             7 : 'Multivitamins +/- minerals', -7 : 'None of the above', -3 : 'Prefer not to answer'},
                   '6179' : {1 : 'Fish oil (including cod liver oil)', 2 : 'Glucosamine', 3 : 'Calcium', 4 :'Zinc', 5 : 'Iron',
                             6 : 'Selenium', -7 : 'None of the above', -3 : 'Prefer not to answer'}
                   }
    cols_numb_onehot = {'6177' : 3, '6153' : 4, '6154' : 6, '6155' : 7, '6179' : 6}
    cols_ordinal = []
    cols_continuous = []
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

    for type_ in ['Cholesterol lowering medication','Blood pressure medication', 'Insulin']:
        df['Medication for cholesterol, blood pressure or diabetes.' + type_] = (df['Medication for cholesterol, blood pressure or diabetes.' + type_] + df['Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones.' + type_])%2
        df = df.drop(columns = ['Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones.' + type_])
    return df


def read_hearing_data(instances = [0, 1, 2, 3], **kwargs):
    """
    2247	Hearing difficulty/problems
    2257	Hearing difficulty/problems with background noise
    3393	Hearing aid user
    4792	Cochlear implant
    4803	Tinnitus
    4814	Tinnitus severity/nuisance => Modify encoding
    4825	Noisy workplace => modify encoding
    4836	Loud music exposure frequency => modify encoding
    ## Remove deaf ppl

    NEW ENCODING :
    13 -> 1
    12 -> 2
    11 -> 3
    """

    dict_onehot = {'4803' : {11 : 'Yes, now most or all of the time', 12 : 'Yes, now a lot of the time', 13 : 'Yes, now some of the time',
                             14 : 'Yes, but not now, but have in the past', 0 :'No, never', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   }
    cols_numb_onehot = {'4803' : 1}
    cols_ordinal = ['2247', '2257', '3393', '4792', '4814', '4825', '4836']
    cols_continuous = []
    cont_fill_na = ['4814']
    cols_half_binary = ['2247', '2257']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)

    ## RE-ENCODE :
    df['Tinnitus severity/nuisance.0'] = df['Tinnitus severity/nuisance.0'].replace(4, 0).replace(13, 1).replace(12, 2).replace(11, 3)
    df['Noisy workplace.0'] = df['Noisy workplace.0'].replace(13, 1).replace(12, 2).replace(11, 3)
    df['Loud music exposure frequency.0'] = df['Loud music exposure frequency.0'].replace(13, 1).replace(12, 2).replace(11, 3)
    return df

# def read_eye_history_data(instance = 0, **kwargs):
#
#     dict_onehot = {'6148' : {1: 'Diabetes related eye disease',
#                              2: 'Glaucoma',
#                              3: 'Injury or trauma resulting in loss of vision',
#                              4: 'Cataract',
#                              5: 'Macular degeneration',
#                              6: 'Other serious eye condition',
#                              -7: 'None of the above'},
#                    '6147' : {1: 'For short-sightedness, i.e. only or mainly for distance viewing such as driving, cinema etc (called myopia)',
#                              2: 'For long-sightedness, i.e. for distance and near, but particularly for near tasks like reading (called hypermetropia)',
#                              3: 'For just reading/near work as you are getting older (called presbyopia)',
#                              4: 'For astigmatism',
#                              5: 'For a squint or turn in an eye since childhood (called strabismus)',
#                              6: 'For a lazy eye or an eye with poor vision since childhood (called amblyopia)',
#                              7: 'Other eye condition'},
#                    '5843' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5832' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5610' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5855' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '6205' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5408' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5877' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5934' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5890' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '6119' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5419' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5441' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                    '5912' : {1 :'Right eye',
#                              2 :'Left eye',
#                              3 :'Both eyes'},
#                 }
#
#     cols_numb_onehot = {'6147' : 6,
#                         '6148' : 5,
#                         '5843' : 1,
#                         '5832' : 1,
#                         '5610' : 1,
#                         '5855' : 1,
#                         '6205' : 1,
#                         '5408' : 1,
#                         '5877' : 1,
#                         '5934' : 1,
#                         '5890' : 1,
#                         '6119' : 1,
#                         '5419' : 1,
#                         '5441' : 1,
#                         '5912' : 1}
#
#
#     cols_onehot = [ key + '-%s.%s' % (instance, int_) for key in cols_numb_onehot.keys() for int_ in range(cols_numb_onehot[key])]
#     cols_ordinal = ['2207', '2217','2227', '5901', '4689', '5430', '4700', '5923', '5945']
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
#     temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continous, **kwargs)
#
#     temp = temp.set_index('eid')
#     temp = temp.dropna(how = 'all')
#
#     display(temp)
#     for column in cols_onehot + cols_ordinal:
#         temp[column] = temp[column].astype('Int64')
#
#     for col in cols_numb_onehot.keys():
#
#         for idx in range(cols_numb_onehot[col]):
#             cate = col + '-%s.%s' % (instance, idx)
#             d = pd.get_dummies(temp[cate])
#             d = d.drop(columns = [ elem for elem in d.columns if int(elem) < 0 ])
#             d.columns = [col + '-%s'%instance + '.' + dict_onehot[col][int(elem)] for elem in d.columns ]
#             temp = temp.drop(columns = [cate])
#
#             if idx == 0:
#                 d_ = d
#             else :
#                 common_cols = d.columns.intersection(d_.columns)
#                 remaining_cols = d.columns.difference(common_cols)
#                 if len(common_cols) > 0 :
#                     d_[common_cols] = d_[common_cols].add(d[common_cols])
#                 for col_ in remaining_cols:
#                     d_[col_] = d[col_]
#         temp = temp.join(d_, how = 'inner')
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
#         split = elem.split('-%s' % instance)
#         features.append(feature_id_to_name[int(split[0])] + split[1])
#     temp.columns = features
#
#
#     return temp
