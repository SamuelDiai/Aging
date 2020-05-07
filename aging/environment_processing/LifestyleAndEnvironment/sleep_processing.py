from ..base_processing import read_complex_data

"""
1160	Sleep duration
1170	Getting up in morning
1180	Morning/evening person (chronotype)
1190	Nap during day
1200	Sleeplessness / insomnia
1210	Snoring
1220	Daytime dozing / sleeping (narcolepsy)
"""

def read_sleep_data(instances = [0, 1, 2, 3], **kwargs):

    dict_onehot = {}
    cols_numb_onehot = {}
    cols_ordinal = ['1170', '1180', '1190', '1200', '1210', '1220']
    cols_continuous = ['1160']
    cont_fill_na = []
    cols_half_binary = {'1180' : 2.5, '1210' : 1.5}


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df

# def read_sleep_data(instance = 0, **kwargs):
#
#     cols_categorical = ['1170','1180','1190',
#             '1200','1210','1220']
#     cols_categorical = [elem + '-%s.0' % instance for elem in cols_categorical]
#     cols_features = ['1160-%s.0' % instance]
#     instance = 0
#     d = read_data(cols_categorical, cols_features, instance, **kwargs)
#     #print(d)
#     bool_remove = (d['Sleep duration.0'] != -1) | (d['Nap during day.0'] != -1) | (d['Getting up in morning.0'] != -1) | (d['Morning/evening person (chronotype).0'] != -1) | (d['Sleeplessness / insomnia.0'] != -1) | (d['Snoring.0'] != -1) | (d['Daytime dozing / sleeping (narcolepsy).0'] != -1) | (d['Sleep duration.0'] != -3) | (d['Nap during day.0'] != -3) | (d['Getting up in morning.0'] != -3) | (d['Morning/evening person (chronotype).0'] != -3) | (d['Sleeplessness / insomnia.0'] != -3) | (d['Snoring.0'] != -3) | (d['Daytime dozing / sleeping (narcolepsy).0'] != -3)
#
#     df = d[bool_remove].dropna(how = 'any')
#     return df
