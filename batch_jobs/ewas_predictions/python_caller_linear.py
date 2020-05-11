import pandas as pd
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_environment_data import load_data, ETHNICITY_COLS
from aging.environment_processing.base_processing import path_output_linear_study

target_dataset = sys.argv[1]
input_dataset = sys.argv[2]

hyperparameters = dict()
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset
print(hyperparameters)


df = load_data(input_dataset, target_dataset).drop(columns = ['eid'])
#df_rescaled, scaler_residual = normalise_dataset(df)
columns_age_sex_ethnicity = ['Age', 'Sex'] + ETHNICITY_COLS
cols_except_age_sex_residual_ethnicty = df.drop(columns = ['residual', 'Age', 'Sex'] + ETHNICITY_COLS).columns


d = pd.DataFrame(columns = ['env_feature_name', 'target_dataset_name', 'p_val', 'corr_value', 'size_na_dropped'])
for column in cols_except_age_sex_residual_ethnicty:
    df_col = df[[column, 'residual'] + columns_age_sex_ethnicity]
    df_col = df_col.dropna()

    lin_residual = LinearRegression()
    lin_residual.fit(df_col[columns_age_sex_ethnicity].values, df_col['residual'].values)
    res_residual = df_col['residual'].values - lin_residual.predict(df_col[columns_age_sex_ethnicity].values)

    lin_feature = LinearRegression()
    lin_feature.fit(df_col[columns_age_sex_ethnicity].values, df_col[column].values)
    res_feature = df_col[column].values - lin_feature.predict(df_col[columns_age_sex_ethnicity].values)

    corr, p_val = pearsonr(res_residual, res_feature)
    d = d.append({'env_feature_name' : column, 'target_dataset_name' : target_dataset, 'p_val' : p_val, 'corr_value' : corr, 'size_na_dropped' : df_col.shape[0]}, ignore_index = True)
d.to_csv(path_output_linear_study + 'linear_correlations_%s_%s.csv' % (input_dataset, target_dataset), index=False)
