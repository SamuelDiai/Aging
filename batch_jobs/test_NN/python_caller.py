import pandas as pd
import os
import sys
test = pd.DataFrame(columns = {'dataset', 'sample_size', 'num features', 'training score'})
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, RandomizedSearchCV, PredefinedSplit, ParameterSampler, cross_validate, train_test_split
from hyperopt import fmin, tpe, space_eval, Trials, hp, STATUS_OK
from sklearn.metrics import r2_score, f1_score
import numpy as np
dataset = sys.argv[1]
architecture = sys.argv[2]
alpha = float(sys.argv[3])

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
params = {
                    'batch_size': 5000,
                    'activation': 'relu', 'max_iter' : 200, 'tol' : 1e-7, 'alpha' : alpha}

hidd_layer = [int(elem) for elem in architecture.split(';')]

params['hidden_layer_sizes'] = hidd_layer


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=3) # 0.25 x 0.8 = 0.2

space = {'alpha': hp.loguniform('alpha', low = np.log(1e-5), high = np.log(1e-1))}
estimator_ = MLPRegressor(**params)
estimator_.fit(X_train.values, y_train.values)

training_score = estimator_.score(X_train.values, y_train.values)
y_test_estimed = estimator_.predict(X_test.values)
scores = r2_score(y_test, y_test_estimed)
# def objective(hyperparameters):
#     estimator_ = MLPRegressor(**params)
#                 ## Set hyperparameters to the model :
#     for key, value in hyperparameters.items():
#         if hasattr(estimator_, key):
#             setattr(estimator_, key, value)
#         else :
#             continue
#     estimator_.fit(X_train.values, y_train.values)
#     y_val_estimed = estimator_.predict(X_val.values)
#     scores = r2_score(y_val, y_val_estimed)
#
#     y_train_estimed = estimator_.predict(X_train.values)
#     score_train = r2_score(y_train, y_train_estimed)
#
#
#     return {'status' : STATUS_OK, 'loss' : -scores, 'attachments' :  {'score_train_val_model_param' :(scores, score_train, estimator_, hyperparameters['alpha'])}}
#
# trials = Trials()
# best = fmin(objective, space, algo = tpe.suggest, max_evals=5, trials = trials)
# best_params = space_eval(space, best)

# d = pd.DataFrame(columns = ['score_val', 'score_train', 'estim', 'alpha'])
# for key, value in trials.attachments.items():
#     score_val, score_train, estim, alpha = value
#     d = d.append({'score_val' : score_val, 'score_train' : score_train, 'estim' : estim, 'alpha' : alpha}, ignore_index = True)
#
#
# score_val_max, score_train_max, estim_max, alpha_max = d.iloc[d['score_val'].idxmax(axis = 1)]
# predict_test = estim_max.predict(X_test.values)
#
# score_test = r2_score(y_test, predict_test)

test = test.append({'dataset' : dataset, 'sample_size' : sample_size, 'num features' : num_features, 'training score' : training_score, 'test score' : scores, 'alpha' : alpha}, ignore_index = True)
test.to_csv('/n/groups/patel/samuel/res_NN/res_2_%s_%s' % (dataset, architecture))


# df_res = pd.DataFrame({'pred' : predict_test, 'real' : y_test})
# df_res.to_csv('/n/groups/patel/samuel/res_NN/pred_2_%s_%s' % (dataset, architecture))
