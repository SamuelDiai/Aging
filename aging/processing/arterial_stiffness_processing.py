from .base_processing import read_data

"""
Features used :
	100007 - Arterial Stiffness
	Errors features : None
	Missing : None
"""

def read_arterial_stiffness_data(**kwargs):
    cols_features =  ['4194-0.0', '4195-0.0',  '4196-0.0', '4198-0.0', '4199-0.0',
                        '4200-0.0',  '4204-0.0', '21021-0.0']
    instance = 0
    cols_filter = []
    return read_data(cols_features, cols_filter, instance, **kwargs)
