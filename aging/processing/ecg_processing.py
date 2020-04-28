from .base_processing import read_data

"""
Features used :
	104 - ECG at rest
	Errors features : '12657'
	Missing : None
"""

def read_ecg_at_rest_data(**kwargs):
	cols_features =  ['12336', '12338', '12340', '22330', '22331', '22332', '22333', '22334', '22335', '22336', '22337', '22338']
	cols_filter = ['12657']
	instance = [2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)
