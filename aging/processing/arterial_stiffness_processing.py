from .base_processing import read_data

"""
Features used :
	100007 - Arterial Stiffness
	Errors features : None
	Missing : None
"""

def read_arterial_stiffness_data(**kwargs):
    cols_features =  ['4194', '4195',  '4196', '4198', '4199',
                        '4200',  '4204', '21021']
    instance = [0, 1, 2, 3]
    cols_filter = []
    return read_data(cols_features, cols_filter, instance, **kwargs)
