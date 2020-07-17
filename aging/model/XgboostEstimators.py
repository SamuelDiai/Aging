
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import numpy as np
class CoxXgboost(BaseEstimator):
    def __init__(self, colsample_bytree=0, gamma=0, learning_rate=0, max_depth=0, n_estimators=0, subsample = 0):
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        #transform y for xgboost formatting
        y_is_dead, y_time = list(zip(*y))
        y_is_dead, y_time = np.array(y_is_dead), np.array(y_time)
        y_time[~y_is_dead] = -y_time[~y_is_dead]
        y = y_time
        dtrain = xgb.DMatrix(X, label = y)
        params = {'objective': 'survival:cox',
                  'colsample_bytree' : self.colsample_bytree,
                  'gamma' : self.gamma,
                  'learning_rate' : self.learning_rate,
                  'max_depth' : self.max_depth,
                  'subsample' : self.subsample}
        print(params, "n_estimators", self.n_estimators)
        bst = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        self.booster_ = bst
        return self

    def predict(self, X):
        try:
            getattr(self, "booster_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        return self.booster_.predict(xgb.DMatrix(X))

    def score(self, X, y):
        HR = self.predict(X)
        y_is_dead, y_time = list(zip(*y))
        return concordance_index_censored(y_is_dead, y_time, HR)[0]

class AftXgboost(BaseEstimator):
    def __init__(self, colsample_bytree=0, gamma=0, learning_rate=0, max_depth=0, n_estimators=0, subsample = 0):
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        #transform y for xgboost formatting
        y_is_dead, y_time = list(zip(*y))
        y_is_dead, y_time = np.array(y_is_dead), np.array(y_time)
        y_lower_bound = y_time
        y_upper_bound = (~y_is_dead) * y_time
        y_upper_bound[(y_upper_bound == 0)] = np.inf
        dtrain = xgb.DMatrix(X)
        dtrain.set_float_info('label_lower_bound', y_lower_bound)
        dtrain.set_float_info('label_upper_bound', y_upper_bound)
        params = {'objective': 'survival:aft',
                  'colsample_bytree' : self.colsample_bytree,
                  'gamma' : self.gamma,
                  'learning_rate' : self.learning_rate,
                  'max_depth' : self.max_depth,
                  'subsample' : self.subsample}
        print(params, "n_estimators", self.n_estimators)
        bst = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        self.booster_ = bst
        return self

    def predict(self, X):
        try:
            getattr(self, "booster_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        return self.booster_.predict(xgb.DMatrix(X))

    def score(self, X, y):
        HR = self.predict(X)
        y_is_dead, y_time = list(zip(*y))
        return concordance_index_censored(y_is_dead, y_time, HR)[0]
