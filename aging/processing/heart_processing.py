from .base_processing import read_data, path_drop
"""
Features used :
	133 - Left ventricular size and function : 22420, 22421, 22422, 22423, 22424, 22425, 22426, 22427
	128 - Pulse wave analysis : 12673, 12674, 12675, 12676, 12677, 12678, 12679, 12680, 
		  12681, 12682, 12683, 12684, 12686, 12687, 12697, 12698, 12699
"""

def read_heart_data(**kwargs):
	cols_features_size = ['22420-2.0', '22421-2.0', '22422-2.0', '22423-2.0', '22424-2.0', '22425-2.0', '22426-2.0', '22427-2.0']
	cols_features_pwa = ['12673-2.0','12674-2.0', '12675-2.0', '12676-2.0', '12677-2.0', '12678-2.0', '12679-2.0', '12680-2.0', 
		  '12681-2.0', '12682-2.0', '12683-2.0', '12684-2.0', '12686-2.0', '12687-2.0', '12697-2.0', '12698-2.0', '12699-2.0']
	cols_filter = []
	cols_features = cols_features_size + cols_features_pwa
	return read_data(cols_features, cols_filter, **kwargs)

def read_heart_size_data(**kwargs):
	cols_features = ['22420-2.0', '22421-2.0', '22422-2.0', '22423-2.0', '22424-2.0', '22425-2.0', '22426-2.0', '22427-2.0']
	cols_filter = []
	return read_data(cols_features, cols_filter, **kwargs)


def read_heart_PWA_data(**kwargs):
	cols_features = ['12673-2.0','12674-2.0', '12675-2.0', '12676-2.0', '12677-2.0', '12678-2.0', '12679-2.0', '12680-2.0', 
		  '12681-2.0', '12682-2.0', '12683-2.0', '12684-2.0', '12686-2.0', '12687-2.0', '12697-2.0', '12698-2.0', '12699-2.0']
	cols_filter = []
	return read_data(cols_features, cols_filter, **kwargs)
