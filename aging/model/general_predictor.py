#!/usr/bin/env python3

import functools
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, StratifiedKFold, RandomizedSearchCV
import numpy as np
import scipy.stats as stat


MODELS = {'ElasticNet', 'RandomForest', 'GradientBoosting', 'Xgboost', 'LightGbm', 'NeuralNetwork'}

class BaseModel():
    def __init__(self, model, outer_splits, inner_splits, n_iter):
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.n_iter = n_iter
        if model not in MODELS:
            raise ValueError(f'{model} model unrecognized')
        else :
            self.model_name = model

    def get_model(self):
        return self.model

    def get_hyper_distribution(self):
        if self.model_name == 'ElasticNet':
            return {
                    'alpha': np.geomspace(0.01, 10, 30),
                    'l1_ratio': stat.uniform(loc = 0.01, scale = 0.99)
                    }
        elif self.model_name == 'RandomForest':
            return {
                    'n_estimators': stat.randint(low = 50, high = 300),
                    'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                    'max_depth': [None, 10, 8, 6]
                    }
        elif self.model_name == 'GradientBoosting':
            return {
                    'n_estimators': stat.randint(low = 250, high = 500),
                    'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                    'learning_rate': stat.uniform(0.01, 0.3),
                    'max_depth': stat.randint(3, 6)
                    }
        elif self.model_name == 'Xgboost':
            return {
                    'colsample_bytree': stat.uniform(loc = 0.2, scale = 0.7),
                    'gamma': stat.uniform(loc = 0.1, scale = 0.5),
                    'learning_rate': stat.uniform(0.02, 0.2),
                    'max_depth': stat.randint(3, 6),
                    'n_estimators': stat.randint(low = 200, high = 400),
                    'subsample': stat.uniform(0.6, 0.4)
            }
        elif self.model_name == 'LightGbm':
            return {
                    'num_leaves': stat.randint(6, 50),
                    'min_child_samples': stat.randint(100, 500),
                    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                    'subsample': stat.uniform(loc=0.2, scale=0.8),
                    'colsample_bytree': stat.uniform(loc=0.4, scale=0.6),
                    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
                }
        elif self.model_name == 'NeuralNetwork':
            return {
                    'learning_rate_init': np.geomspace(5e-5, 2e-2, 30),
                    'alpha': np.geomspace(1e-5, 1e-1, 30),
                    'hidden_layer_sizes': [(100, 50), (30, 10)],
                    'batch_size': [8, 32],
                    'activation': ['tanh', 'relu']
            }


    def optimize_hyperparameters_fold_(self, X, y, index, scoring, fold):
        outer_cv = KFold(n_splits = self.outer_splits, shuffle = False, random_state = 0)
        index_train, index_test = list(outer_cv.split(X, y))[fold]
        X_train, X_test, y_train, y_test = X[index_train], X[index_test], y[index_train], y[index_test]

        inner_cv = KFold(n_splits = self.inner_splits, random_state = 0)
        clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = inner_cv, n_jobs = -1, scoring = scoring, verbose = 10, n_iter = self.n_iter)

        clf.fit(X_train, y_train)

        best_estim = clf.best_estimator_
        best_params = clf.best_params_
        best_params_flat = []
        for elem in best_params.values():
    	    if type(elem) == tuple:
                for sub_elem in elem:
            	    best_params_flat.append(sub_elem)
    	    else:
                best_params_flat.append(elem)
        self.best_params = best_params_flat

        y_predict_test = best_estim.predict(X_test)
        y_predict_train = best_estim.predict(X_train)

        eid_test = index[index_test]
        eid_train = index[index_train]

        df_test = pd.DataFrame(data = {'eid' : eid_test, 'fold' : fold, 'predictions' : y_predict_test, 'real' : y_test} )
        df_train = pd.DataFrame(data = {'eid' : eid_train, 'fold' : fold, 'predictions' : y_predict_train, 'real' : y_train} )

        return df_test, df_train



    def features_importance_(self, X, y, scoring):
        cv = KFold(n_splits = self.inner_splits, shuffle = False)
        clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = cv, n_jobs = -1, scoring = scoring, n_iter = self.n_iter)
        clf.fit(X, y)
        best_estim = clf.best_estimator_

        if self.model_name == 'ElasticNet':
            self.features_imp = np.abs(best_estim.coef_) / np.sum(np.abs(best_estim.coef_))
        elif self.model_name == 'RandomForest':
            self.features_imp = best_estim.feature_importances_
        elif self.model_name == 'GradientBoosting':
            self.features_imp = best_estim.feature_importances_
        elif self.model_name == 'Xgboost':
            self.features_imp = best_estim.feature_importances_
        elif self.model_name == 'LightGbm':
            self.features_imp = best_estim.feature_importances_ / np.sum(best_estim.feature_importances_)
        elif self.model_name == 'NeuralNetwork':
            raise ValueError('No feature_importances for NN')
        else :
            raise ValueError('Wrong model name')
