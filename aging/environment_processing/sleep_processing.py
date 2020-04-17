from .base_processing import read_data

"""
Features used :
    51428 - Infectious Diseases :
    1307 - Infectious Disease Antigens :

    Errors features :
    Missing :
"""

def read_sleep_data(instance = 0, **kwargs):

    cols_categorical = ['1170','1180','1190',
            '1200','1210','1220']
    cols_categorical = [elem + '-%s.0' % instance for elem in cols_categorical]
    cols_features = ['1160-%s.0' % instance]
    instance = 0
    d = read_data(cols_categorical, cols_features, instance, **kwargs)
    #print(d)
    bool_remove = (d['Sleep duration.0'] != -1) | (d['Nap during day.0'] != -1) | (d['Getting up in morning.0'] != -1) | (d['Morning/evening person (chronotype).0'] != -1) | (d['Sleeplessness / insomnia.0'] != -1) | (d['Snoring.0'] != -1) | (d['Daytime dozing / sleeping (narcolepsy).0'] != -1) | (d['Sleep duration.0'] != -3) | (d['Nap during day.0'] != -3) | (d['Getting up in morning.0'] != -3) | (d['Morning/evening person (chronotype).0'] != -3) | (d['Sleeplessness / insomnia.0'] != -3) | (d['Snoring.0'] != -3) | (d['Daytime dozing / sleeping (narcolepsy).0'] != -3)

    df = d[bool_remove].dropna(how = 'any')
    return df
