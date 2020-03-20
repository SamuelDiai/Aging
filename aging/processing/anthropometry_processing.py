from .base_processing import read_data

"""
Features used :
	100009 - Impedance measurements
	Errors features : None
	Missing : 23105
"""

def read_anthropometry_impedance_data(**kwargs):
    cols_features = [
    '23098-0.0', '23099-0.0', '23100-0.0', '23101-0.0', '23102-0.0', '23104-0.0',
    '23106-0.0', '23107-0.0', '23108-0.0', '23109-0.0', '23110-0.0', '23111-0.0',
    '23112-0.0', '23113-0.0', '23114-0.0', '23115-0.0', '23116-0.0', '23117-0.0',
    '23118-0.0', '23119-0.0', '23120-0.0', '23121-0.0', '23122-0.0', '23123-0.0',
    '23124-0.0', '23125-0.0', '23126-0.0', '23127-0.0', '23128-0.0', '23129-0.0', '23130-0.0' ]
    cols_filter = []
    return read_data(cols_features, cols_filter, **kwargs)
