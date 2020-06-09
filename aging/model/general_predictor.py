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
import copy
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, RandomizedSearchCV, PredefinedSplit, ParameterSampler
import numpy as np
import scipy.stats as stat
from sklearn.metrics import r2_score, f1_score
from hyperopt import fmin, tpe, space_eval, Trials, hp, STATUS_OK



MODELS = {'ElasticNet', 'RandomForest', 'GradientBoosting', 'Xgboost', 'LightGbm', 'NeuralNetwork'}




class BaseModel():
    def __init__(self, model, outer_splits, inner_splits, n_iter, model_validate = 'HyperOpt'):
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.n_iter = n_iter
        self.model_validate = model_validate
        if model not in MODELS:
            raise ValueError(f'{model} model unrecognized')
        else :
            self.model_name = model

    def get_model(self):
        return self.model

    def get_hyper_distribution(self):
        if self.model_validate == 'HyperOpt':
            if self.model_name == 'ElasticNet':
                return {
                        'alpha':  hp.loguniform('alpha', low = np.log(0.01), high = np.log(10)),
                        'l1_ratio': hp.uniform('l1_ratio', low = 0.01, high = 0.99)
                        }
            elif self.model_name == 'RandomForest':
                return {
                        'n_estimators': hp.randint('n_estimators', upper = 300) + 150,
                        'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
                        'max_depth': hp.choice('max_depth', [None, 10, 8, 6])
                        }
            elif self.model_name == 'GradientBoosting':
                return {
                        'n_estimators': hp.randint('n_estimators', upper = 300) + 150,
                        'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                        'learning_rate': hp.uniform('learning_rate', low = 0.01, high = 0.3),
                        'max_depth': hp.randint('max_depth', 10) + 5
                        }
            elif self.model_name == 'Xgboost':
                return {
                        'colsample_bytree': hp.uniform('colsample_bytree', low = 0.2, high = 0.7),
                        'gamma': hp.uniform('gamma', low = 0.1, high = 0.5),
                        'learning_rate': hp.uniform('learning_rate', low = 0.02, high = 0.2),
                        'max_depth': hp.randint('max_depth', 10) + 5,
                        'n_estimators': hp.randint('n_estimators', 300) + 150,
                        'subsample': hp.uniform('subsample', 0.6, 0.4)
                }
            elif self.model_name == 'LightGbm':
                return {
                        'num_leaves': hp.randint('num_leaves', 40) + 5,
                        'min_child_samples': hp.randint('min_child_samples', 400) + 100,
                        'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]),
                        'subsample': hp.uniform('subsample', low=0.2, high=0.8),
                        'colsample_bytree': hp.uniform('colsample_bytree', low=0.4, high=0.6),
                        'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
                        'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100]),
                        'n_estimators' : hp.randint('n_estimators', 300) + 150
                    }
            elif self.model_name == 'NeuralNetwork':
                return {
                        'learning_rate_init': hp.loguniform('learning_rate_init', low = np.log(5e-5), high = np.log(2e-2)),
                        'alpha': hp.loguniform('alpha', low = np.log(1e-5), high = np.log(1e-1)),
                        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100, 50), (30, 10)]),
                        'batch_size': hp.choice('batch_size', [1000, 500]),
                        'activation': hp.choice('activation', ['tanh', 'relu'])
                }
        elif self.model_validate == 'RandomizedSearch':
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
                        'batch_size': [1000, 500],
                        'activation': ['tanh', 'relu']
                }



    def optimize_hyperparameters_fold_(self, X, y, scoring, fold):
        """
        input X  : dataframe with features + eid
        input y : dataframe with target + eid
        """

        if self.inner_splits != self.outer_splits - 1:
            raise ValueError('n_inner_splits should be equal to n_outer_splits - 1 ! ')


        X_eid = X.drop_duplicates('eid')
        y_eid = y.drop_duplicates('eid')
        #
        outer_cv = KFold(n_splits = self.outer_splits, shuffle = False, random_state = 0)
        #
        #        # if outer_splits = 10, split 1/10 for testing and 9/10 for training
        #         # test_fold contain all test list_test_folds
        #         # train_fold contain all the input dataset except the test fold
        #
        list_test_folds = [elem[1] for elem in outer_cv.split(X_eid, y_eid)]
        list_train_folds =  list(outer_cv.split(X_eid, y_eid))[fold][0]
        #
        #
        list_test_folds_eid = [X_eid.eid[elem].values for elem in list_test_folds]
        list_train_folds_eid = X_eid.eid[list_train_folds].values
        #
        list_train_fold_id = X.index[X.eid.isin(list_train_folds_eid)]
        list_test_folds_id = [X.index[X.eid.isin(list_test_folds_eid[elem])].values for elem in range(len(list_test_folds_eid))]
        #
        index_train, index_test = list_train_fold_id, list_test_folds_id[fold]
        list_test_folds_id = list_test_folds_id[:fold] + list_test_folds_id[fold + 1 :]
        X = X.drop(columns = ['eid'])
        y = y.drop(columns =['eid'])
        print(X.columns, y.columns)
        X_train, X_test, y_train, y_test = X.loc[index_train], X.loc[index_test], y.loc[index_train], y.loc[index_test]

        ## Create custom Splits
        list_test_folds_id_index = [np.array([X_train.index.get_loc(elem) for elem in list_test_folds_id[fold_num]]) for fold_num in range(len(list_test_folds_id))]
        test_folds = np.zeros(len(X_train), dtype = 'int')
        for fold_count in range(len(list_test_folds_id)):
            test_folds[list_test_folds_id_index[fold_count]] = fold_count
        inner_cv = PredefinedSplit(test_fold = test_folds)
        #
        ## RandomizedSearch :
        if self.model_validate == 'RandomizedSearch':
            clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = inner_cv, n_jobs = -1, scoring = scoring, verbose = 10, n_iter = self.n_iter, return_train_score = False)
            clf.fit(X_train.values, y_train.values)
            best_estim = copy.deepcopy(clf.best_estimator_)
            best_params = copy.deepcopy(clf.best_params_)
            results = clf.cv_results_
            results = pd.DataFrame(data = results)

            params_per_fold_opt = results.params[results[['split%s_test_score' % elem for elem in range(self.inner_splits)]].idxmax()]
            params_per_fold_opt = dict(params_per_fold_opt.reset_index(drop = True))

        ## HyperOpt :
        elif self.model_validate == 'HyperOpt':
            def objective(hyperparameters):
                estimator_ = self.get_model()
                ## Set hyperparameters to the model :
                for key, value in hyperparameters.items():
                    if hasattr(estimator_, key):
                        setattr(estimator_, key, value)
                    else :
                        continue
                scores = cross_validate(estimator_, X_train.values, y_train.values, scoring = scoring, cv = inner_cv, verbose = 10 )
                return {'status' : STATUS_OK, 'loss' : scores['test_score'].mean(), 'attachments' : {'split_test_scores' : scores['test_score']}}
            space = self.get_hyper_distribution()
            trials = Trials()
            best = fmin(objective, space, algo = tpe.suggest, max_evals=self.n_iter, trials = trials)
            best_params = space_eval(space, best)
            ## Recreate best estim :
            estim = self.get_model()
            for key, value in best_params.items():
                if hasattr(estim, key):
                    setattr(estim, key, value)
                else :
                    continue
            best_estim = estim.fit(X_train.values, y_train.values)
            results = pd.DataFrame(columns = ['params'] + ['split%s_test_score' % elem for elem in range(self.inner_splits)]) # nsplits
            for idx, value in trials.attachments.items():
                d = dict()
                for split in range(self.inner_splits):
                    d['split%s_test_score' % split] = value[0][split]
                d['params'] = value[1]
                results = results.append(d, ignore_index = True)
                params_per_fold_opt = results.params[results[['split%s_test_score' % elem for elem in range(self.inner_splits)]].idxmax()]
                params_per_fold_opt = dict(params_per_fold_opt.reset_index(drop = True))


        list_train_val = []
        list_train_train = []
        for inner_fold in range(self.inner_splits):
            index_train_val = list_test_folds_id[inner_fold]
            index_train_train = [elem for elem in index_train if elem not in index_train_val]
            X_train_train, X_train_val, y_train_train, y_train_val = X.loc[index_train_train], X.loc[index_train_val], y.loc[index_train_train], y.loc[index_train_val]

            model_ = self.get_model()
            for key, value in params_per_fold_opt[inner_fold].items():
                if hasattr(model_, key):
                    setattr(model_, key, value)
                else :
                    continue
            model_.fit(X_train_train.values, y_train_train.values)

            y_predict_train_val_fold = model_.predict(X_train_val.values)

            df_train_val = pd.DataFrame(data = {'id' : index_train_val, 'outer_fold' : np.nan, 'pred' : y_predict_train_val_fold })
            list_train_val.append(df_train_val)


        df_val = pd.concat(list_train_val)

        y_predict_test = best_estim.predict(X_test)
        y_predict_train = best_estim.predict(X_train)
        df_test = pd.DataFrame(data = {'id' : index_test, 'outer_fold' : fold, 'pred' : y_predict_test} )
        df_train = pd.DataFrame(data = {'id' : index_train, 'outer_fold' : fold, 'pred' : y_predict_train })


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
        columns = X.columns
        y = y.values
        if False :
        #if self.model_name != 'NeuralNetwork':
            X = X.values
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
            else :
                raise ValueError('Wrong model name')
        else :
            list_scores = []
            get_init_hyper = list(ParameterSampler(self.get_hyper_distribution(), n_iter = 1))[0]
            estimator = self.get_model()
            for index, value in get_init_hyper.items():
                setattr(estimator, index, value)
            estimator.fit(X.values, y)
            if scoring == 'r2':
                score_max = r2_score(y, estimator.predict(X.values))
            else :
                score_max = f1_score(y, estimator.predict(X.values))
            for column in columns :
                X_copy = copy.deepcopy(X)
                X_copy[column] = np.random.permutation(X_copy[column])
                #estimator.fit(X_copy.values, y)
                if scoring == 'r2':
                    score = r2_score(y, estimator.predict(X_copy.values))
                else :
                    score = f1_score(y, estimator.predict(X_copy.values))
                list_scores.append(score_max - score)
            self.features_imp = list_scores
