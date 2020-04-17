from .base_processing import path_data, path_dictionary



def read_physical_activity_data(instance = 0, **kwargs):
#instance = 2
    array_length = 5

    cols_physical_types = ['6164-%s.%s'% (instance, val) for val in range(array_length)]
    matching_physical = [981, 971, 3647, 3637, 1001, 991, 1021, 1011, 2634, 2624, 894, 914, 874]
    matching_physical = [str(elem) + '-%s.0' % instance for elem in matching_physical]
    cols_moderate = ['884-%s.0' % instance]
    cols_vigorous = ['904-%s.0' % instance ]
    cols_walking = ['864-%s.0' % instance]
    df = pd.read_csv(path_data , usecols = matching_physical + cols_walking + cols_vigorous + cols_moderate + cols_physical_types + ['eid'], nrows = None
                    ).set_index('eid')
    for column in (cols_physical_types + cols_walking + cols_vigorous+ cols_moderate + cols_physical_types):
        df[column] = df[column].astype('Int64')

    df2 = df.dropna(how = 'all')
    df2 = df2[(df2 >= 0)]
    df2 = df2.replace(np.nan, 0)
    df2 = df2[matching_physical + cols_moderate + cols_vigorous+ cols_walking]


    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']


    features_index = df2.columns
    features = []
    for elem in features_index:
        print(elem)
        features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
    df2.columns = features
    return df2
