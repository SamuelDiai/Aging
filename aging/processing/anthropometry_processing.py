from .base_processing import read_data

"""
Features used :
	100009 - Impedance measurements
	100010 - BodySize
	Errors features : None
	Missing : 23105
"""

def read_anthropometry_impedance_data(**kwargs):

	"""
	23113	Leg fat-free mass (right)
	23118	Leg predicted mass (left)
	23114	Leg predicted mass (right)
	23123	Arm fat percentage (left)
	23119	Arm fat percentage (right)
	23124	Arm fat mass (left)
	23120	Arm fat mass (right)
	23121	Arm fat-free mass (right)
	23125	Arm fat-free mass (left)
	23126	Arm predicted mass (left)
	23122	Arm predicted mass (right)
	23127	Trunk fat percentage
	23128	Trunk fat mass
	23129	Trunk fat-free mass
	23130	Trunk predicted mass
	# 23105	Basal metabolic rate => Not taken : Function of the Age
	23099	Body fat percentage
	23100	Whole body fat mass
	23101	Whole body fat-free mass
	23102	Whole body water mass
	23115	Leg fat percentage (left)
	23111	Leg fat percentage (right)
	23116	Leg fat mass (left)
	23112	Leg fat mass (right)
	23117	Leg fat-free mass (left)
	23106	Impedance of whole body
	23110	Impedance of arm (left)
	23109	Impedance of arm (right)
	23108	Impedance of leg (left)
	23107	Impedance of leg (right)
	"""
	cols_features = ['23099', '23100', '23101', '23102',
    '23106', '23107', '23108', '23109', '23110', '23111',
    '23112', '23113', '23114', '23115', '23116', '23117',
    '23118', '23119', '23120', '23121', '23122', '23123',
    '23124', '23125', '23126', '23127', '23128', '23129', '23130' ]
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)


def read_anthropometry_body_size_data(**kwargs):
	"""

	48	Waist circumference
	21002	Weight
	21001	Body mass index (BMI)
	49	Hip circumference
	50	Standing height
	20015	Seatting height
	"""
	cols_features = ['48', '21002', '21001', '49', '50', '20015']
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_anthropometry_data(**kwargs):
	cols_features_body = ['48', '21002', '21001', '49', '50', '51']
	cols_features_imp = ['23099', '23100', '23101', '23102',
    '23106', '23107', '23108', '23109', '23110', '23111',
    '23112', '23113', '23114', '23115', '23116', '23117',
    '23118', '23119', '23120', '23121', '23122', '23123',
    '23124', '23125', '23126', '23127', '23128', '23129', '23130' ]
	cols_features = cols_features_imp + cols_features_body
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)
