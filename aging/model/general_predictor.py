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

from ..processing.abdominal_composition_processing import read_abdominal_data
from ..processing.brain_processing import read_grey_matter_volumes_data, read_subcortical_volumes_data, read_brain_data
from ..processing.heart_processing import read_heart_data, read_heart_size_data, read_heart_PWA_data
from ..processing.body_composition_processing import read_body_composition_data
from ..processing.bone_composition_processing import read_bone_composition_data


NAME = {'ElasticNet', 'RandomForest', 'GradientBoosting', 'Xgboost', 'LightGbm', 'NeuralNetwork'}

dataset_to_field = {'AbdominalComposition' : 149, 
                    'BrainGreyMatter' : 1101,
                    'BrainSubcorticalVolumes': 1102,
                    'Brain' : 100,
                    'Heart' : 102,
                    'HeartSize' : 133,
                    'HeartPWA' : 128,
                    'BodyComposition' : 124,
                    'BoneComposition' : 125}

class BaseModel():
    def __init__(self, name, outer_splits, inner_splits, n_iter):
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.n_iter = n_iter
        if name not in NAME:
            raise ValueError(f'{name} model unrecognized')
        else :
            self.name = name      
                
    def fit(self, df):
        return self.model.fit(X, y)

    def get_model(self):
        return self.model

    def get_hyper_distribution(self):
        if self.name == 'ElasticNet':
            return {
                    'alpha': np.geomspace(0.01, 10, 30),
                    'l1_ratio': stat.uniform(loc = 0.01, scale = 0.99)
                    }
        elif self.name == 'RandomForest':
            return {
                    'n_estimators': stat.randint(low = 50, high = 300),
                    'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                    'max_depth': [None, 10, 8, 6]
                    }
        elif self.name == 'GradientBoosting':
            return {
                    'n_estimators': stat.randint(low = 250, high = 500),
                    'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                    'learning_rate': stat.uniform(0.01, 0.3),
                    'max_depth': stat.randint(3, 6)
                    }
        elif self.name == 'Xgboost':
            return {
                    'colsample_bytree': stat.uniform(loc = 0.2, scale = 0.7),
                    'gamma': stat.uniform(loc = 0.1, scale = 0.5),
                    'learning_rate': stat.uniform(0.02, 0.2),
                    'max_depth': stat.randint(3, 6),
                    'n_estimators': stat.randint(low = 200, high = 400),
                    'subsample': stat.uniform(0.6, 0.4)
            }
        elif self.name == 'LightGbm':
            return {
                    'num_leaves': stat.randint(6, 50), 
                    'min_child_samples': stat.randint(100, 500), 
                    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                    'subsample': stat.uniform(loc=0.2, scale=0.8), 
                    'colsample_bytree': stat.uniform(loc=0.4, scale=0.6),
                    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
                }
        elif self.name == 'NeuralNetwork':
            return {
                    'learning_rate_init': np.geomspace(5e-5, 2e-2, 30),
                    'alpha': np.geomspace(1e-5, 1e-1, 30),
                    'hidden_layer_sizes': [(100, 50), (30, 10)],
                    'batch_size': [8, 32],
                    'activation': ['tanh', 'relu']
            }


         
    def optimize_hyperparameters_(self, X, y, scoring):

        list_predicts_test_fold = []
        list_index_test_fold = []
        outer_cv = StratifiedKFold(n_splits = self.outer_splits)
        for index_train, index_test in outer_cv.split(X, y):
            
            X_train, X_test, y_train, y_test = X[index_train], X[index_test], y[index_train], y[index_test]

            inner_cv = KFold(n_splits = self.inner_splits, shuffle = False, random_state = 0)
            clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = inner_cv, n_jobs = -1, scoring = scoring)
            clf.fit(X_train, y_train)

            best_estim = clf.best_estimator_
            y_predict = best_estim.predict(X_test)
            list_predicts_test_fold.append(y_predict)
            list_index_test_fold.append(index_test)
            


        concat_predicts = np.concatenate(list_predicts_test_fold)
        concat_index = np.concatenate(list_index_test_fold)
        concat_index_order = concat_index.argsort()
        concat_predicts_ordered = concat_predicts[concat_index_order]

    def optimize_hyperparameters_fold_(self, X, y, index, scoring, fold):
        outer_cv = KFold(n_splits = self.outer_splits, shuffle = False, random_state = 0)
        index_train, index_test = list(outer_cv.split(X, y))[fold]
        X_train, X_test, y_train, y_test = X[index_train], X[index_test], y[index_train], y[index_test]

        inner_cv = KFold(n_splits = self.inner_splits, shuffle = False, random_state = 0)
        clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = inner_cv, n_jobs = -1, scoring = scoring, verbose = 10)

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
       


    def features_importance_(self, X, y, n_splits, n_iter, scoring):
        cv = KFold(n_splits = n_splits, shuffle = False, random_state = 0)
        clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = cv, n_jobs = -1, scoring = scoring)
        clf.fit(X, y)
        best_estim = clf.best_estimator_

        if self.name == 'ElasticNet':
            self.features_imp = np.abs(best_estim.coef_) / np.sum(np.abs(best_estim.coef_))
        elif self.name == 'RandomForest':
            self.features_imp = best_estim.feature_importances_
        elif self.name == 'GradientBoosting':
            self.features_imp = best_estim.feature_importances_
        elif self.name == 'Xgboost':
            self.features_imp = best_estim.feature_importances_
        elif self.name == 'LightGbm':
            self.features_imp = best_estim.feature_importances_ / np.sum(best_estim.feature_importances_)
        elif self.name == 'NeuralNetwork':
            raise ValueError('No feature_importances for NN')
        else :
            raise ValueError('Wrong model name')





    
class GeneralPredictor(BaseModel):
    def __init__(self, name, outer_splits, inner_splits, n_iter, target, dataset):
        BaseModel.__init__(self, name, outer_splits, inner_splits, n_iter)
        self.name = name
        self.dataset = dataset
        if target == 'Sex': 
            self.scoring = 'f1'
            self.target = 'Sex'
            if name == 'ElasticNet':
                self.model = ElasticNet(max_iter = 2000)
            elif name == 'RandomForest':
                self.model = RandomForestClassifier()
            elif name == 'GradientBoosting':
                self.model = GradientBoostingClassifier()
            elif name == 'Xgboost':
                self.model = XGBClassifier()
            elif name == 'LightGbm':
                self.model = LGBMClassifier()
            elif name == 'NeuralNetwork':
                self.model = MLPClassifier(solver = 'adam')
        elif target == 'Age':
            self.scoring = 'r2'
            self.target = 'Age'
            if name == 'ElasticNet':
                self.model = ElasticNet(max_iter = 2000)
            elif name == 'RandomForest':
                self.model = RandomForestRegressor()
            elif name == 'GradientBoosting':
                self.model = GradientBoostingRegressor()
            elif name == 'Xgboost':
                self.model = XGBRegressor()
            elif name == 'LightGbm':
                self.model = LGBMRegressor()
            elif name == 'NeuralNetwork':
                self.model = MLPRegressor(solver = 'adam')
        else :
            raise ValueError('target : "%s" not valid, please enter "Sex" or "Age"' % target)

    def feature_importance(self, df, n_iter, n_splits):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
            y = df['Sex'].values
        elif self.target == 'Age':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
            y = df['Age when attended assessment centre'].values
        else :
            raise ValueError('GeneralPredictor not instancied')

        self.features_importance_(X, y, n_splits, n_iter, self.scoring)
        final_df = pd.DataFrame(data = {'features' : df.drop(columns = ['Sex', 'Age when attended assessment centre']).columns, 'weight' : self.features_imp})
        final_df.set_index('features').to_csv('/n/groups/patel/samuel/Aging/aging/feature_importances/FeatureImp_' + self.target + '_' + self.dataset + '_' + self.name + '.csv')

        
    def optimize_hyperparameters_fold(self, df, fold):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
            y = df['Sex'].values
        elif self.target == 'Age':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
            y = df['Age when attended assessment centre'].values
        else :
            raise ValueError('GeneralPredictor not instancied')
        return self.optimize_hyperparameters_fold_(X, y, df.index, self.scoring, fold)

    
    def load_dataset(self, **kwargs):
        nrows = None
        if 'nrows' in kwargs.keys():
            nrows = kwargs['nrows']
        if self.dataset not in dataset_to_field.keys():
            raise ValueError('Wrong dataset name ! ')
        else :
            self.field_id = dataset_to_field[self.dataset]
            if self.dataset == 'AbdominalComposition':
                df = read_abdominal_data(**kwargs)
            elif self.dataset == 'Brain':
                df = read_brain_data(**kwargs)
            elif self.dataset == 'BrainGreyMatterVolumes':
                df = read_grey_matter_volumes_data(**kwargs)
            elif self.dataset == 'BrainSubCorticalVolumes':
                df = read_subcortical_volumes_data(**kwargs)
            elif self.dataset == 'Heart':
                df = read_heart_data(**kwargs)
            elif self.dataset == 'HeartSize':
                df = read_heart_size_data(**kwargs) 
            elif self.dataset == 'HeartPWA':
                df = read_heart_PWA_data(**kwargs) 
            elif self.dataset == 'BoneComposition':
                df = read_bone_composition_data(**kwargs)
            elif self.dataset == 'BodyComposition':
                df = read_body_composition_data(**kwargs)
            return df  


    def save_predictions(self, predicts_df, step, fold):
        hyper_parameters_name = '_'.join([str(elem) for elem in self.best_params])
        if len(self.best_params) != 7:
            hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(self.best_params))])

        filename = 'Predictions_' + self.target + '_' + self.dataset + '_' + str(dataset_to_field[self.dataset]) + '_main' +  '_raw' + '_' + self.name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '_B.csv'
        predicts_df.to_csv('/n/groups/patel/samuel/Aging/aging/predictions/' + filename)

    def normalise_dataset(self, df):
        # Save old data
        df_without_sex = df.drop(columns = 'Sex')
        sex_column = df['Sex']
        cols = df_without_sex.columns
        indexes = df_without_sex.index

        # save scaler
        scaler = StandardScaler()
        scaler.fit(df_without_sex)

        scaler_age = StandardScaler()
        scaler_age.fit(df['Age when attended assessment centre'].values.reshape(-1, 1))
        self.scaler = scaler_age

        # Scale and create Dataframe
        array_rescaled =  scaler.transform(df_without_sex)
        df_rescaled = pd.DataFrame(array_rescaled, columns = cols, index = indexes).join(sex_column)

        return df_rescaled

    def inverse_normalise_dataset(self, df_rescaled):
        
        if self.target == 'Sex':
            return df_rescaled
        elif self.target == 'Age':
            df_noscaled = df_rescaled
            if hasattr(self, 'scaler'):
                df_noscaled['predictions'] = self.scaler.inverse_transform(df_noscaled['predictions'].values.reshape(-1, 1))
                df_noscaled['real'] = self.scaler.inverse_transform(df_noscaled['real'].values.reshape(-1, 1))
            return df_noscaled
        else :
            raise ValueError('dataframe is not rescaled') 






