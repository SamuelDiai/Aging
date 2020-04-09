#!/usr/bin/env python3

import functools
import numpy as np
import pandas as pd
import copy

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

        if self.inner_splits != self.outer_splits - 1:
            raise ValueError('n_inner_splits should be equal to n_outer_splits - 1 ! ')

        # Define OUTER FOLD
        outer_cv = KFold(n_splits = self.outer_splits, shuffle = False, random_state = 0)

        # if outer_splits = 10, split 1/10 for testing and 9/10 for training
        # test_fold contain all test list_test_folds
        # train_fold contain all the input dataset except the test fold

        list_test_folds = [elem[1] for elem in outer_cv.split(X, y)]
        list_train_folds =  list(outer_cv.split(X, y))[fold][0]


        index_train, index_test = list_train_folds, list_test_folds[fold]
        list_test_folds = list_test_folds[:fold] + list_test_folds[fold + 1 :]

        X_train, X_test, y_train, y_test = X[index_train], X[index_test], y[index_train], y[index_test]

        # Define inner cv
        inner_cv = KFold(n_splits = self.inner_splits, random_state = 0)
        clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = inner_cv, n_jobs = -1, scoring = scoring, verbose = 10, n_iter = self.n_iter)

        # Fit on the (9/10)*(4/5)
        clf.fit(X_train, y_train)

        best_estim = copy.deepcopy(clf.best_estimator_)
        best_params = copy.deepcopy(clf.best_params_)
        results = clf.cv_results_
        results = pd.DataFrame(data = results)

        params_per_fold_opt = results.params[results[['split%s_test_score' % elem for elem in range(self.inner_splits)]].idxmax()]
        params_per_fold_opt = dict(params_per_fold_opt.reset_index(drop = True))

        list_train_val = []
        list_train_train = []
        for inner_fold in range(self.inner_splits):
            index_train_val = list_test_folds[inner_fold]
            index_train_train = [elem for elem in index_train if elem not in index_train_val]
            X_train_train, X_train_val, y_train_train, y_train_val = X[index_train_train], X[index_train_val], y[index_train_train], y[index_train_val]

            model_ = self.get_model()
            for key, value in params_per_fold_opt[inner_fold].items():
                if hasattr(model_, key):
                    setattr(model_, key, value)
                else :
                    continue
            model_.fit(X_train_train, y_train_train)

            y_predict_train_val_fold = model_.predict(X_train_val)
            y_predict_train_train_fold = model_.predict(X_train_train)

            eid_train_val = index[index_train_val]
            eid_train_train = index[index_train_train]

            df_train_val = pd.DataFrame(data = {'eid' : eid_train_val, 'fold' : fold, 'predictions' : y_predict_train_val_fold })
            df_train_train = pd.DataFrame(data = {'eid' : eid_train_train, 'fold' : fold, 'predictions' : y_predict_train_train_fold })
            list_train_val.append(df_train_val)
            list_train_train.append(df_train_train)

        df_val = pd.concat(list_train_val)
        df_train = pd.concat(list_train_train)


        y_predict_test = best_estim.predict(X_test)
        eid_test = index[index_test]
        df_test = pd.DataFrame(data = {'eid' : eid_test, 'fold' : fold, 'predictions' : y_predict_test} )
        #df_train = pd.DataFrame(data = {'eid' : eid_train, 'fold' : fold, 'predictions' : y_predict_train, 'real' : y_train} )

        # return model_.predict(X_train_train), model_.predict(X_train_val)
        #
        # print("Results : ", clf.cv_results_ )
        #
        #
        #
        # print("CPU COUNT : ", mp.cpu_count())
        # import time
        #
        # # t1 = time.time()
        # #
        # func = partial(generate_train_and_val_preds, X_train = X_train, y_train = y_train, model_ = self.get_model(), best_params = best_params, inner_cv = inner_cv)
        # # test = Parallel(n_jobs=-1)(delayed(func)(inner_fold) for inner_fold in range(self.inner_splits))
        # # t2 = time.time()
        # # print(test)
        # # print("JOBLIB : ", t2 - t1)
        #
        #
        # # t3 = time.time()
        # #
        # # for inner_fold in range(self.inner_splits):
        # #     print(func(inner_fold))
        # # t4 = time.time()
        # #
        # # print("Classic : ", t4 - t3)
        #
        #
        #
        # pool = mp.Pool(mp.cpu_count())
        #
        # test = pool.imap_unordered(func, range(self.inner_splits))
        # for elem in test:
        #     print(elem)
        # #pool.close()
        # #pool.join()
        # print(test.get())




        best_params_flat = []
        for elem in best_params.values():
    	    if type(elem) == tuple:
                for sub_elem in elem:
            	    best_params_flat.append(sub_elem)
    	    else:
                best_params_flat.append(elem)
        self.best_params = best_params_flat



        return df_test, df_val, df_train



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
