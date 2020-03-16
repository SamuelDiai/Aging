from .base_processing import read_data

"""
Features used :
	124 - Body Composition
	Errors features : None
	Missing : None
"""

def read_body_composition_data(**kwargs):
	cols_features = [str(elem) + '-2.0' for elem in range(23244, 23289 + 1)]
	cols_filter = [ ]
	return read_data(cols_features, cols_filter, **kwargs)

