from .base_processing import read_data

"""
Features used :
	100011 - Blood Pressure
	Errors features : None
	Missing : None
"""

def read_blood_pressure_data(**kwargs):
    cols_features = ['102-0.0', '102-0.1', '4079-0.0', '4079-0.1', '4080-0.0', '4080-0.1']
    cols_filter = []
    instance = 0
    temp = read_data(cols_features, cols_filter, instance, **kwargs)
    for elem in pd.Series([elem.split('.')[0] for elem in temp.columns.values if 'Age' not in elem and 'Sex' not in elem]).drop_duplicates():
        temp[elem + '.0'] = (t[elem + '.0'] + t[elem + '.1'])/2
    return temp[[elem for elem in t.columns if '.1' not in elem]]
