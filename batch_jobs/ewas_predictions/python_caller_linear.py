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

target_dataset = sys.argv[1]
input_dataset = sys.argv[2]


hyperparameters = dict()
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset
print(hyperparameters)



# def normalise_dataset(df):
#
#     scaler_residual = StandardScaler()
#     scaler_residual.fit(df['residual'].values.reshape(-1, 1))
#
# # Get categorical data apart from continous ones
#     df_cat = df.select_dtypes(include=['int'])
#     df_cont = df.drop(columns = df_cat.columns)
#
#     cols = df_cont.columns
#     indexes = df_cont.index
#
#     # save scaler
#     scaler = StandardScaler()
#     scaler.fit(df_cont)
#
#     # Scale and create Dataframe
#     array_rescaled =  scaler.transform(df_cont)
#     df_rescaled = pd.DataFrame(array_rescaled, columns = cols, index = indexes).join(df_cat)
#
#     return df_rescaled, scaler_residual

df = load_data(input_dataset, target_dataset, nrows = 20000).drop(columns = ['eid'])
#df_rescaled, scaler_residual = normalise_dataset(df)
columns_age_sex_ethnicity = ['Age', 'Sex'] + ethnicity_cols
cols_except_age_sex_residual_ethnicty = df.drop(columns = ['residual', 'Age', 'Sex'] + ethnicity_cols).columns


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
d.to_csv('/n/groups/patel/samuel/EWAS/linear_output_v2/linear_correlations_%s_%s.csv' % (input_dataset, target_dataset), index=False)



#
#
#
#
#
# lin_residual = LinearRegression()
# lin_residual.fit(df[['Age', 'Sex'] + ].values, df_rescaled['residual'].values)
# res_residual = df_rescaled['residual'].values - lin_residual.predict(df[['Age', 'Sex']].values)
#
#
#
#
# lin_residual = LinearRegression()
# columns_age_sex_ethnicity = ['Age', 'Sex'] + list(df_ethicity.columns)
# lin_residual.fit(df_all[columns_age_sex_ethnicity].values, df_all['residual'].values)
# res_residual = df_all['residual'].values - lin_residual.predict(df_all[columns_age_sex_ethnicity].values)
#
# for column in df_diagnose.columns:
#     lin_feature = LinearRegression()
#     lin_feature.fit(df_all[columns_age_sex_ethnicity].values, df_all[column].values)
#     res_feature = df_all[column].values - lin_feature.predict(df_all[columns_age_sex_ethnicity].values)
#
#     corr, p_val = pearsonr(res_residual, res_feature)
#     d = d.append({'env_feature_name' : column, 'target_dataset_name' : target_dataset, 'p_val' : p_val, 'corr_value' : corr}, ignore_index = True)
#
#
#
#
#
# for column in cols_except_age_sex_residual:
#     lin_feature = LinearRegression()
#     lin_feature.fit(df_rescaled[['Age', 'Sex']].values, df_rescaled[column].values)
#     res_feature = df_rescaled[column].values - lin_feature.predict(df_rescaled[['Age', 'Sex']].values)
#
#     corr, p_val = pearsonr(res_residual, res_feature)
#     d = d.append({'env_feature_name' : column, 'target_dataset_name' : target_dataset, 'p_val' : p_val, 'corr_value' : corr}, ignore_index = True)
#
#
