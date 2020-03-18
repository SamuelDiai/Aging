from .base_processing import read_data

"""
Features used :
	125 - Bone composition
	Errors features : None
	Missing : 23207, 23294, 23303, 23211
"""

def read_bone_composition_data(**kwargs):
	missing =  ['23207-2.0', '23294-2.0', '23303-2.0', '23211-2.0']
	cols_features = [str(elem) + '-2.0' for elem in range(23200, 23243 + 1)] + [str(elem) + '-2.0' for elem in range(23290, 23318 + 1)] + ['23320-2.0']
	cols_filter = [ ]

	return read_data([elem for elem in cols_features if elem not in missing], cols_filter, **kwargs)

