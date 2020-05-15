import pandas as pd
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, linregress

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_environment_data import load_data, ETHNICITY_COLS, load_data_env
from aging.environment_processing.base_processing import path_output_linear_study

target_dataset = sys.argv[1]
input_dataset = sys.argv[2]

hyperparameters = dict()
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset
print(hyperparameters)


# df = load_data(input_dataset, target_dataset).drop(columns = ['eid'])
#
#
#
# ## Linear EWAS :
# #df_rescaled, scaler_residual = normalise_dataset(df)
# columns_age_sex_ethnicity = ['Age', 'Sex'] + ETHNICITY_COLS
# cols_except_age_sex_residual_ethnicty = df.drop(columns = ['residual', 'Age', 'Sex'] + ETHNICITY_COLS).columns
#
#
# d = pd.DataFrame(columns = ['env_feature_name', 'target_dataset_name', 'p_val', 'corr_value', 'size_na_dropped'])
# for column in cols_except_age_sex_residual_ethnicty:
#     df_col = df[[column, 'residual'] + columns_age_sex_ethnicity]
#     df_col = df_col.dropna()
#
#     lin_residual = LinearRegression()
#     lin_residual.fit(df_col[columns_age_sex_ethnicity].values, df_col['residual'].values)
#     res_residual = df_col['residual'].values - lin_residual.predict(df_col[columns_age_sex_ethnicity].values)
#
#     lin_feature = LinearRegression()
#     lin_feature.fit(df_col[columns_age_sex_ethnicity].values, df_col[column].values)
#     res_feature = df_col[column].values - lin_feature.predict(df_col[columns_age_sex_ethnicity].values)
#
#     corr, p_val = pearsonr(res_residual, res_feature)
#     d = d.append({'env_feature_name' : column, 'target_dataset_name' : target_dataset, 'p_val' : p_val, 'corr_value' : corr, 'size_na_dropped' : df_col.shape[0]}, ignore_index = True)
# d.to_csv(path_output_linear_study + 'linear_correlations_%s_%s.csv' % (input_dataset, target_dataset), index=False)
#
#
# ## See effect of Age per feature
# d2 = pd.DataFrame(columns = ['target_dataset_name', 'env_feature_name', 'p_val', 'r_val', 'cslope'])
#
# columns_sex_ethnicity = ['Sex'] + ETHNICITY_COLS
# for column in cols_except_age_sex_residual_ethnicty:
#     df_col2 = df[[column] + columns_age_sex_ethnicity]
#     df_col2 = df_col2.dropna()
#
#     lin_residual2 = LinearRegression()
#     lin_residual2.fit(df_col2[columns_sex_ethnicity].values, df_col2['Age'].values)
#     res_residual2 = df_col2['Age'].values - lin_residual2.predict(df_col2[columns_sex_ethnicity].values)
#
#     lin_feature2 = LinearRegression()
#     lin_feature2.fit(df_col2[columns_sex_ethnicity].values, df_col2[column].values)
#     res_feature2 = df_col2[column].values - lin_feature2.predict(df_col2[columns_sex_ethnicity].values)
#
#     cslope, intercept, r_value, p_value, std_err = linregress(res_feature2, res_residual2)
#     d2 = d2.append({'target_dataset_name' : target_dataset,'env_name' : input_dataset, 'env_feature_name' : column,   'p_val' : p_value, 'r_val' : r_value, 'cslope' : cslope, 'intercept' : intercept}, ignore_index = True)
# d2.to_csv('/n/groups/patel/samuel/EWAS/LinearAge/' +  'linear_age_%s_%s.csv' % (input_dataset, target_dataset), index = False)



## Compute dissimilarity :
df_env = load_data_env(input_dataset).drop(columns = ['eid'])
df_target = load_data_env(target_dataset).drop(columns = ['eid'])
large_join = df_env.join(df_target, how = 'outer', lsuffix = '_l', rsuffix = '_r')
del df_env
del df_target
large_join_shape = large_join.shape[0]
tiny_join = large_join.dropna()
tiny_join_shape = tiny_join.shape[0]


if large_join_shape == 0:
    quotient = 0
else :
    quotient =  tiny_join_shape / large_join_shape


df_res = pd.DataFrame({'Target Dataset': [target_dataset],  'Environmental Dataset' : [input_dataset], 'Intersection' : [tiny_join_shape], 'Union' : [large_join_shape], 'Dissimilartiy': [quotient]})
df_res.to_csv('/n/groups/patel/samuel/EWAS/SampleSizes.csv', mode = 'a', index = False)
