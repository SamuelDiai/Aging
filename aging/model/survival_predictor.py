from .general_predictor import *
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from .load_and_save_survival_data import load_data_survival
from .XgboostEstimators import CoxXgboost

class SurvivalPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, dataset, fold, model_validate = 'HyperOpt'):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter, model_validate)
        self.fold = fold
        self.dataset = dataset
        self.scoring = None
        if model == 'CoxPh':
            self.model = CoxPHSurvivalAnalysis()
        elif model == 'CoxRf':
            self.model = RandomSurvivalForest()
        elif model == 'CoxGbm':
            self.model = GradientBoostingSurvivalAnalysis()
        elif model == 'CoxXgboost':
            self.model = CoxXgboost()
        else :
            raise ValueError('model : "%s" not valid, please enter "CoxPh" or "CoxRf" or "CoxGbm"' % model)

    def set_organ_view(self, organ, view):
        self.organ = organ
        self.view = view

    def load_dataset(self, **kwargs):
        return load_data_survival(self.dataset, **kwargs)

    def optimize_hyperparameters_fold(self, df):
        X = df.drop(columns = ['y'])
        y = df[['y', 'eid']]
        return self.optimize_hyperparameters_fold_(X, y, self.scoring, self.fold, self.organ)

    def feature_importance(self, df):
        X = df.drop(columns = ['y'])
        y = df[['y', 'eid']]
        self.features_importance_(X, y, self.scoring, self.organ)
        return df.drop(columns = ['y', 'eid']).columns
