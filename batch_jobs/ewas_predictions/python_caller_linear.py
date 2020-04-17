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

from aging.model.load_and_save_environment_data import load_data

target_dataset = sys.argv[1]
input_dataset = sys.argv[2]


hyperparameters = dict()
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset
print(hyperparameters)

def normalise_dataset(df):

    scaler_residual = StandardScaler()
    scaler_residual.fit(df['residual'].values.reshape(-1, 1))

# Get categorical data apart from continous ones
    df_cat = df.select_dtypes(include=['int'])
    df_cont = df.drop(columns = df_cat.columns)

    cols = df_cont.columns
    indexes = df_cont.index

    # save scaler
    scaler = StandardScaler()
    scaler.fit(df_cont)

    # Scale and create Dataframe
    array_rescaled =  scaler.transform(df_cont)
    df_rescaled = pd.DataFrame(array_rescaled, columns = cols, index = indexes).join(df_cat)

    return df_rescaled, scaler_residual


df = load_data(input_dataset, target_dataset)
df_rescaled, scaler_residual = normalise_dataset(df)
cols_except_age_sex_residual = df_rescaled.drop(columns = ['residual', 'Age', 'Sex']).columns


d = pd.DataFrame(columns = ['env_feature_name', 'target_dataset_name', 'p_val', 'corr_value'])
lin_residual = LinearRegression()
lin_residual.fit(df_rescaled[['Age', 'Sex']].values, df_rescaled['residual'].values)
res_residual = df_rescaled['residual'].values - lin_residual.predict(df[['Age', 'Sex']].values)

for column in cols_except_age_sex_residual:
    lin_feature = LinearRegression()
    lin_feature.fit(df_rescaled[['Age', 'Sex']].values, df_rescaled[column].values)
    res_feature = df_rescaled[column].values - lin_feature.predict(df_rescaled[['Age', 'Sex']].values)

    corr, p_val = pearsonr(res_residual, res_feature)
    d = d.append({'env_feature_name' : column, 'target_dataset_name' : target_dataset, 'p_val' : p_val, 'corr_value' : corr}, ignore_index = True)

d.to_csv('/n/groups/patel/samuel/EWAS/linear_output/linear_correlations_%s_%s.csv' % (input_dataset, target_dataset), index=False)
