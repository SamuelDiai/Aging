import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_environment_data import load_data

target_dataset = sys.argv[1]
input_dataset = sys.argv[2]


hyperparameters = dict()
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset
print(hyperparameters)

df = load_data(input_dataset, target_dataset)
cols_except_age_sex_residual = df.drop(columns = ['residual', 'Age when attended assessment centre', 'Sex']).columns
col_age = df['Age when attended assessment centre']
col_sex = df['Sex']
col_residual = df['residual']

d = pd.DataFrame(columns = ['env_feature_name', 'target_dataset_name', 'p_val', 'corr_value'])
for column in cols_except_age_sex_residual:
    lin_residual = LinearRegression()
    lin_residual.fit(df[['Age when attended assessment centre', 'Sex']].values, df['residual'].values)
    res_residual = lin_residual.predict(df[['Age when attended assessment centre', 'Sex']].values) - df['residual'].values

    lin_feature = LinearRegression()
    lin_feature.fit(df[['Age when attended assessment centre', 'Sex']].values, df[column].values)
    res_feature = lin_feature.predict(df[['Age when attended assessment centre', 'Sex']].values) - df[column].values

    corr, p_val = pearsonr(res_residual, res_feature)
    d = d.append({'env_feature_name' : column, 'target_dataset_name' : target_dataset, 'p_val' : p_val, 'corr_value' : corr}, ignore_index = True)

d.to_csv('/n/groups/patel/samuel/EWAS/linear_output/linear_correlations_%s_%s' % (input_dataset, target_dataset))
