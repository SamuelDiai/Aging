import sys
import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.environment_processing.base_processing import path_input_env, path_input_env_inputed, path_inputs_env
from aging.model.InputtingNans import compute_linear_coefficients_for_each_col, load_raw_data, input_variables_in_column


cols_ethnicity = ['Do_not_know', 'Prefer_not_to_answer', 'NA', 'White', 'British',
       'Irish', 'White_Other', 'Mixed', 'White_and_Black_Caribbean',
       'White_and_Black_African', 'White_and_Asian', 'Mixed_Other', 'Asian',
       'Indian', 'Pakistani', 'Bangladeshi', 'Asian_Other', 'Black',
       'Caribbean', 'African', 'Black_Other', 'Chinese', 'Other_ethnicity',
       'Other']
cols_age_sex_eid_ethnicity = ['Sex', 'eid', 'Age when attented assessment centre'] + cols_ethnicity


n_cores = int(sys.argv[1])

## Load Full raw data
list_int_cols, continuous_cols, final_df = load_raw_data(path_raw = path_input_env, path_output = path_input_env_inputed, path_inputs = path_inputs_env)
col_age_id_eid_sex_ethnicty = final_df[cols_age_sex_eid_ethnicity]
## split continuous and categorical
split_cols_continuous = np.array_split(continuous_cols, n_cores)
split_cols_categorical = np.array_split(list_int_cols, n_cores)


def parallel_group_of_features(split_col, categorical):
    list_features_split = []
    for col in split_col:
        column_modified = compute_coefs_and_input(col, categorical)
        list_features_split.append(column_modified[col])

    inputed_res = col_age_id_eid_sex_ethnicty
    for column in list_features_split:
        inputed_res = inputed_res.join(column)
    return inputed_res


pool = Pool(n_cores)
final_df_inputed_non_cate = pool.map(lambda split_col  : parallel_group_of_features(split_col, categorical = False), split_cols_continuous)
pool.close()
pool.join()


pool2 = Pool(n_cores)
final_df_inputed_cate = pool.map(lambda split_col  : parallel_group_of_features(split_col, categorical = True), split_cols_categorical)
pool2.close()
pool2.join()


final_df_inputed = col_age_id_eid_sex_ethnicty
for df in final_df_inputed_non_cate :
     final_df_inputed = final_df_inputed.join(df, how = 'outer', rsuffix = '_r')
     final_df_inputed = final_df_inputed[final_df_inputed.columns[~final_df_inputed.columns.str.contains('_r')]]

for df in final_df_inputed_cate:
    final_df_inputed = final_df_inputed.join(df, how = 'outer', rsuffix = '_r')
    final_df_inputed = final_df_inputed[final_df_inputed.columns[~final_df_inputed.columns.str.contains('_r')]]


final_df_inputed.to_csv(path_input_env_inputed)
