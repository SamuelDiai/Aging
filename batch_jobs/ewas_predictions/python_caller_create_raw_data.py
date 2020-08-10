import sys
import os
import glob
import pandas as pd

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.environment_processing.base_processing import path_inputs_env, path_input_env, ETHNICITY_COLS
from aging.model.load_and_save_environment_data import map_envdataset_to_dataloader_and_field


df_sex_age_ethnicity = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv').set_index('id')

for idx, df in enumerate(map_envdataset_to_dataloader_and_field.keys()):
    print(df)
    path = path_inputs_env + df + '.csv'
#for idx, path in enumerate(glob.glob(path_inputs_env + '*.csv')):
    df_temp = pd.read_csv(path).set_index('id')
    used_cols = [elem for elem in df_temp.columns if elem not in ['Sex', 'eid', 'Age when attended assessment centre'] + ETHNICITY_COLS]
    int_cols = df_temp.select_dtypes(int).columns
    df_temp[int_cols] = df_temp[int_cols].astype('Int64')
    if idx == 0:
        final_df = df_sex_age_ethnicity.join(df_temp[used_cols], how = 'outer')
    else :
        final_df = final_df.join(df_temp[used_cols], how = 'outer')
print("Starting merge")
final_df = final_df.reset_index().merge(df_ethnicity, on = 'eid').set_index('id')
final_df.to_csv(path_input_env)
