from .base_processing import read_data

"""
Features used :
	1102 - Subcortical volumes (FIRST) : 25011 -> 25024
	1101 - Regional grey matter volumes (FAST)  : 25782 -> 25920
	Errors features : None
	Missing : None
"""

def read_grey_matter_volumes_data(**kwargs):
	cols_features = [str(elem) + '-2.0' for elem in range(25782, 25920 + 1)]
	cols_filter = [ ]
	return read_data(cols_features, cols_filter, **kwargs)

def read_subcortical_volumes_data(**kwargs):
	cols_features = [str(elem) + '-2.0' for elem in range(25011, 25024 + 1)]
	cols_filter = [ ]
	return read_data(cols_features, cols_filter, **kwargs)

def read_brain_data(**kwargs):
	cols_features = [str(elem) + '-2.0' for elem in range(25011, 25024 + 1)] + [str(elem) + '-2.0' for elem in range(25782, 25920 + 1)]
	cols_filter = [ ]
	return read_data(cols_features, cols_filter, **kwargs)