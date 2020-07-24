from .general_predictor import *
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from .load_and_save_survival_data import load_data_survival, load_data_survivalregression, save_predictions_regression_to_csv
from .XgboostEstimators import CoxXgboost, AftXgboost

class SurvivalPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, target, dataset, fold, model_validate = 'HyperOpt'):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter, model_validate)
        self.fold = fold
        self.dataset = dataset
        self.scoring = None
        self.target = target
        if model == 'CoxPh':
            self.model = CoxPHSurvivalAnalysis()
        elif model == 'CoxRf':
            self.model = RandomSurvivalForest()
        elif model == 'CoxGbm':
            self.model = GradientBoostingSurvivalAnalysis()
        elif model == 'CoxXgboost':
            self.model = CoxXgboost()
        elif model == 'AftXgboost':
            self.model = AftXgboost()
        else :
            raise ValueError('model : "%s" not valid, please enter "CoxPh" or "CoxRf" or "CoxGbm"' % model)

    def set_organ_view(self, organ, view):
        self.organ = organ
        self.view = view

    def load_dataset(self, **kwargs):
        return load_data_survival(self.dataset, self.target, **kwargs)

    def optimize_hyperparameters_fold(self, df):
        X = df.drop(columns = ['y'])
        y = df[['y', 'eid']]
        return self.optimize_hyperparameters_fold_(X, y, self.scoring, self.fold, self.organ, view = None)

    def feature_importance(self, df):
        X = df.drop(columns = ['y'])
        y = df[['y', 'eid']]
        self.features_importance_(X, y, self.scoring, self.organ, view = None)
        return df.drop(columns = ['y', 'eid']).columns

    def save_predictions(self, predicts_df, step):
        if 'Cluster' in self.dataset:
            dataset_proper = self.dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
        else :
            dataset_proper = self.dataset
        if not hasattr(self, 'best_params'):
            raise ValueError('Predictions not trained')
        save_predictions_regression_to_csv(predicts_df, step, self.target, dataset_proper, self.model_name, self.fold, self.best_params)


class SurvivalRegressionPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, target, dataset, fold, model_validate = 'HyperOpt'):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter, model_validate)
        self.dataset = dataset
        self.fold = fold
        self.scoring = 'r2'
        self.target = target
        if model == 'ElasticNet':
            self.model = ElasticNet(max_iter = 2000)
        elif model == 'RandomForest':
            self.model = RandomForestRegressor()
        elif model == 'GradientBoosting':
            self.model = GradientBoostingRegressor()
        elif model == 'Xgboost':
            self.model = XGBRegressor()
        elif model == 'LightGbm':
            self.model = LGBMRegressor()
        elif model == 'NeuralNetwork':
            self.model = MLPRegressor(solver = 'adam', activation = 'relu', hidden_layer_sizes = (128, 64, 32), batch_size = 1000, early_stopping = True)
        else :
            raise ValueError('model : "%s" not valid, please enter "ElasticNet" or "LightGbm" or "NeuralNetwork"' % model)

    def set_organ_view(self, organ, view):
        self.organ = organ
        self.view = view

    def load_dataset(self, **kwargs):
        return load_data_survivalregression(self.dataset, self.target, **kwargs)

    def optimize_hyperparameters_fold(self, df):
        X = df.drop(columns = ['Follow up time'])
        y = df[['Follow up time', 'eid']]
        return self.optimize_hyperparameters_fold_(X, y, self.scoring, self.fold, self.organ, view = self.view)

    def feature_importance(self, df):
        X = df.drop(columns = ['Follow up time'])
        y = df[['Follow up time', 'eid']]
        self.features_importance_(X, y, self.scoring, self.organ, view = self.view)
        return df.drop(columns = ['Follow up time', 'eid']).columns

    def save_features(self, cols):
        if 'Cluster' in self.dataset:
            dataset_proper = self.dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
        else :
            dataset_proper = self.dataset
        if not hasattr(self, 'features_imp') and self.model_name != 'Correlation' :
            raise ValueError('Features importance not trained')
        save_features_to_csv(cols, self.features_imp, self.target, self.organ, self.view, self.model_name, method = None)
        save_features_to_csv(cols, self.features_imp_sd, self.target, self.organ, self.view, self.model_name, method = 'sd')
        save_features_to_csv(cols, self.features_imp_mean, self.target, self.organ, self.view, self.model_name, method = 'mean')

    def save_predictions(self, predicts_df, step):
        if 'Cluster' in self.dataset:
            dataset_proper = self.dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
        else :
            dataset_proper = self.dataset
        if not hasattr(self, 'best_params'):
            raise ValueError('Predictions not trained')
        save_predictions_regression_to_csv(predicts_df, step, self.target, dataset_proper, self.model_name, self.fold, self.best_params)
