import pandas as pd
import os
test = pd.DataFrame(columns = {'dataset', 'sample_size', 'num features', 'training score'})
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler


dataset = sys.argv[1]
architecture = sys.argv[2]

df_path = '/n/groups/patel/samuel/save_final_inputs/' + dataset + '.csv'

def normalise_dataset(df):
    # Save old data
    df_without_sex_and_eid = df.drop(columns = ['Sex', 'eid'])
    sex_and_eid_columns = df[['Sex', 'eid']]
    cols = df_without_sex_and_eid.columns
    indexes = df_without_sex_and_eid.index

    # save scaler
    scaler = StandardScaler()
    scaler.fit(df_without_sex_and_eid)

    scaler_age = StandardScaler()
    scaler_age.fit(df['Age when attended assessment centre'].values.reshape(-1, 1))


    # Scale and create Dataframe
    array_rescaled =  scaler.transform(df_without_sex_and_eid)
    df_rescaled = pd.DataFrame(array_rescaled, columns = cols, index = indexes).join(sex_and_eid_columns)

    return df_rescaled, scaler_age



df_ = pd.read_csv(df_path).set_index('id').dropna()
sample_size = df_.shape[0]

df_ethnicity = pd.read_csv('/n/groups/patel/samuel/ethnicities.csv').set_index('eid')
df_with_ethnicity = df_.reset_index().merge(df_ethnicity, on = 'eid').set_index('id')
num_features = df_with_ethnicity.shape[1]


df_rescaled, scaler_age = normalise_dataset(df_with_ethnicity)
X = df_rescaled.drop(columns = ['eid', 'Age when attended assessment centre'])
y = df_rescaled['Age when attended assessment centre']
params = { 'alpha': 0,
                    'batch_size': 5000,
                    'activation': 'relu', 'max_iter' : 200, 'tol' : 1e-7}

hidd_layer = [int(elem) for elem in architecture.split(';')]

params['hidden_layer_sizes'] = hidd_layer
c = MLPRegressor(**params)
c.fit(X.values, y.values)
score = c.score(X.values, y.values)

test = test.append({'dataset' : dataset, 'sample_size' : sample_size, 'num features' : num_features, 'training score' : score}, ignore_index = True)
test.to_csv('/n/groups/patel/samuel/res_NN/res_%s_%s' % (dataset, architecture))

prediction = c.predict(X.values)
real = y.values
df_res = pd.DataFrame({'pred' : prediction, 'real' : real})
df_res.to_csv('/n/groups/patel/samuel/res_NN/pred_%s_%s' % (dataset, architecture))
