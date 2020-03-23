from .base_processing import read_data

"""
Features used :
	104 - ECG at rest
	Errors features : '12657-2.0'
	Missing : None
"""

def read_ecg_at_rest_data(**kwargs):
	cols_features =  ['12336-2.0', '12338-2.0', '12340-2.0', '22330-2.0', '22331-2.0', '22332-2.0', '22333-2.0', '22334-2.0', '22335-2.0', '22336-2.0', '22337-2.0', '22338-2.0']
	cols_filter = ['12657-2.0']
	instance = 2
	return read_data(cols_features, cols_filter, instance, **kwargs)
