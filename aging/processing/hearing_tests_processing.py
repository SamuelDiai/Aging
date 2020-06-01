from .base_processing import read_data



"""
Hearing Test
20019	Speech-reception-threshold (SRT) estimate (left)
20021	Speech-reception-threshold (SRT) estimate (right)
"""

def read_hearing_test_data(**kwargs):

    cols_features = [4277, 4270, 20019, 20021]
    cols_filter = []
    instance = [0, 1, 2, 3]
    return read_data(cols_features, cols_filter, instance, **kwargs)
