from .base_processing import read_data

"""
Features used :
    51428 - Infectious Diseases :
    1307 - Infectious Disease Antigens :

    Errors features :
    Missing :
"""

def read_infectious_diseases_data(**kwargs):
    cols_categorical = []
    cols_features = ['23050-0.0','23051-0.0','23052-0.0','23053-0.0','23054-0.0','23055-0.0','23056-0.0','23057-0.0','23058-0.0',
 '23059-0.0','23060-0.0','23061-0.0','23062-0.0','23063-0.0','23064-0.0','23065-0.0','23066-0.0','23067-0.0','23068-0.0' '23069-0.0','23070-0.0','23074-0.0','23075-0.0']
    instance = 0

    return read_data(cols_categorical, cols_features, instance, **kwargs)


def read_infectious_disease_antigens_data(**kwargs):
    cols_categorical = []
    cols_features = [ ]
    instance = 0

    return read_data(cols_categorical, cols_features, instance, **kwargs)
