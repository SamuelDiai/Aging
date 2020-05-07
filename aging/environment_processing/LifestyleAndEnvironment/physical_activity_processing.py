from ..base_processing import read_complex_data

"""
884     Number of days/week of moderate physical activity 10+ minutes
904     Number of days/week of vigorous physical activity 10+ minutes
864     Number of days/week walked 10+ minutes

6164    Types of physical activity in last 4 weeks
6162    Types of transport used (excluding work)

2634    Duration of heavy DIY => fill na
1021    Duration of light DIY => fill na
3647    Duration of other exercises => fill na
1001    Duration of strenuous sports => fill na
894     Duration of moderate activity => fill na
914     Duration of vigorous activity => fill na
874     Duration of walks => fill na
981     Duration walking for pleasure => fill na
2624    Frequency of heavy DIY in last 4 weeks => fill na
1011    Frequency of light DIY in last 4 weeks => fill na
3637    Frequency of other exercises in last 4 weeks => fill na
943     Frequency of stair climbing in last 4 weeks => fill na
991     Frequency of strenuous sports in last 4 weeks => fill na
971     Frequency of walking for pleasure in last 4 weeks => fill na
924     Usual walking pace => fill na

1100	Drive faster than motorway speed limit
1090	Time spent driving
1080	Time spent using computer
1070	Time spent watching television (TV)

"""

def read_physical_activity_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'6164' : {1 : 'Walking for pleasure (not as a means of transport)', 2 : 'Other exercises (eg: swimming, cycling, keep fit, bowling)', 3 : 'Strenuous sports', 4 : 'Light DIY (eg: pruning, watering the lawn)',
                             5 : 'Heavy DIY (eg: weeding, lawn mowing, carpentry, digging)', -7 : 'None of the above', -3 : 'Prefer not to answer'},
                   '6162' : {1 : 'Car/motor vehicle', 2 : 'Walk', 3 : 'Public transport', 4 : 'Cycle', -7 : 'None of the above', -3 : 'Prefer not to answer'},
                  }

    cols_numb_onehot = {'6164' : 5, '6162' : 4}
    cols_ordinal = ['884', '904', '864',
                    '2634','1021','3647','1001','894','914', '874', '981','2624','1011','3637','943','991','971','924',
                    '1100', '1090', '1080', '1070']
    cols_continuous = []
    cont_fill_na = ['2634','1021','3647','1001','894','914', '874', '981','2624','1011','3637','943','991','971','924']
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
    df['Drive faster than motorway speed limit.0'] = df['Drive faster than motorway speed limit.0'].replace(5, 0)

    return df

#
# def read_physical_activity_data(instance = 0, **kwargs):
# #instance = 2
#     array_length = 5
#
#     cols_physical_types = ['6164-%s.%s'% (instance, val) for val in range(array_length)]
#     matching_physical = [981, 971, 3647, 3637, 1001, 991, 1021, 1011, 2634, 2624, 894, 914, 874]
#     matching_physical = [str(elem) + '-%s.0' % instance for elem in matching_physical]
#     cols_moderate = ['884-%s.0' % instance]
#     cols_vigorous = ['904-%s.0' % instance ]
#     cols_walking = ['864-%s.0' % instance]
#     df = pd.read_csv(path_data , usecols = matching_physical + cols_walking + cols_vigorous + cols_moderate + cols_physical_types + ['eid'], nrows = None
#                     ).set_index('eid')
#     for column in (cols_physical_types + cols_walking + cols_vigorous+ cols_moderate + cols_physical_types):
#         df[column] = df[column].astype('Int64')
#
#     df2 = df.dropna(how = 'all')
#     df2 = df2[(df2 >= 0)]
#     df2 = df2.replace(np.nan, 0)
#     df2 = df2[matching_physical + cols_moderate + cols_vigorous+ cols_walking]
#
#
#     df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
#     df_features.set_index('FieldID', inplace = True)
#     feature_id_to_name = df_features.to_dict()['Field']
#
#
#     features_index = df2.columns
#     features = []
#     for elem in features_index:
#         print(elem)
#         features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
#     df2.columns = features
#     return df2
