from .base_processing import read_data

"""
Features used :
	100009 - Impedance measurements
	100010 - BodySize
	Errors features : None
	Missing : 23105
"""

def read_anthropometry_impedance_data(**kwargs):
	cols_features = [
    '23098', '23099', '23100', '23101', '23102', '23104',
    '23106', '23107', '23108', '23109', '23110', '23111',
    '23112', '23113', '23114', '23115', '23116', '23117',
    '23118', '23119', '23120', '23121', '23122', '23123',
    '23124', '23125', '23126', '23127', '23128', '23129', '23130' ]
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)


def read_anthropometry_body_size_data(**kwargs):
	cols_features = ['48', '21002', '21001', '49', '50', '51', '20015']
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_anthropometry_data(**kwargs):
	cols_features_body = ['48', '21002', '21001', '49', '50', '51', '20015']
	cols_features_imp = [
    '23098', '23099', '23100', '23101', '23102', '23104',
    '23106', '23107', '23108', '23109', '23110', '23111',
    '23112', '23113', '23114', '23115', '23116', '23117',
    '23118', '23119', '23120', '23121', '23122', '23123',
    '23124', '23125', '23126', '23127', '23128', '23129', '23130' ]
	cols_features = cols_features_imp + cols_features_body
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)
