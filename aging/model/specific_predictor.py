
from .general_predictor import *
from .load_and_save_data import load_data, save_features_to_csv, save_predictions_to_csv, dict_dataset_to_organ_and_view


class GeneralPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, target, dataset, fold, model_validate = 'HyperOpt'):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter, model_validate)
        self.fold = fold
        self.dataset = dataset
        if target == 'Sex':
            self.scoring = 'f1'
            self.target = 'Sex'
            if model == 'ElasticNet':
                self.model = SGDClassifier(loss = 'log', penalty = 'elasticnet', max_iter = 2000)
            elif model == 'RandomForest':
                self.model = RandomForestClassifier()
            elif model == 'GradientBoosting':
                self.model = GradientBoostingClassifier()
            elif model == 'Xgboost':
                self.model = XGBClassifier()
            elif model == 'LightGbm':
                self.model = LGBMClassifier()
            elif model == 'NeuralNetwork':
                self.model = MLPClassifier(solver = 'adam', activation = 'relu', hidden_layer_sizes = (128, 64, 32), batch_size = 1000, early_stopping = True)
        elif target == 'Age':
            self.scoring = 'r2'
            self.target = 'Age'
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
            raise ValueError('target : "%s" not valid, please enter "Sex" or "Age"' % target)

    def set_organ_view(self, organ, view):
        self.organ = organ
        self.view = view

    def load_dataset(self, **kwargs):
        return load_data(self.dataset, **kwargs)


    def optimize_hyperparameters_fold(self, df):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre'])
            y = df[['Sex', 'eid']]
        elif self.target == 'Age':
            X = df.drop(columns = ['Age when attended assessment centre'])
            y = df[['Age when attended assessment centre', 'eid']]
        else :
            raise ValueError('GeneralPredictor not instancied')
        return self.optimize_hyperparameters_fold_(X, y, self.scoring, self.fold, self.organ)


    def feature_importance(self, df):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre'])
            y = df[['Sex', 'eid']]
            self.features_importance_(X, y, self.scoring, self.organ)
            return df.drop(columns = ['Sex', 'Age when attended assessment centre', 'eid']).columns
        elif self.target == 'Age':
            X = df.drop(columns = ['Age when attended assessment centre'])
            y = df[['Age when attended assessment centre', 'eid']]
            self.features_importance_(X, y, self.scoring, self.organ)
            return df.drop(columns = ['Age when attended assessment centre', 'eid']).columns
        else :
            raise ValueError('GeneralPredictor not instancied')

    # def feature_importance(self, df):
    #     if self.target == 'Sex':
    #         X = df.drop(columns = ['Sex', 'Age when attended assessment centre', 'eid'])
    #         y = df['Sex']
    #         self.features_importance_(X, y, self.scoring, self.organ)
    #         return df.drop(columns = ['Sex', 'Age when attended assessment centre', 'eid']).columns
    #     elif self.target == 'Age':
    #         X = df.drop(columns = ['Age when attended assessment centre', 'eid'])
    #         y = df['Age when attended assessment centre']
    #         self.features_importance_(X, y, self.scoring, self.organ)
    #         return df.drop(columns = ['Age when attended assessment centre', 'eid']).columns
    #     else :
    #         raise ValueError('GeneralPredictor not instancied')


    # def normalise_dataset(self, df):
    #     # Save old data
    #     df_without_sex_and_eid = df.drop(columns = ['Sex', 'eid'])
    #     sex_and_eid_columns = df[['Sex', 'eid']]
    #     cols = df_without_sex_and_eid.columns
    #     indexes = df_without_sex_and_eid.index
    #
    #     # save scaler
    #     scaler = StandardScaler()
    #     scaler.fit(df_without_sex_and_eid)
    #
    #     scaler_age = StandardScaler()
    #     scaler_age.fit(df['Age when attended assessment centre'].values.reshape(-1, 1))
    #     self.scaler = scaler_age
    #
    #     # Scale and create Dataframe
    #     array_rescaled =  scaler.transform(df_without_sex_and_eid)
    #     df_rescaled = pd.DataFrame(array_rescaled, columns = cols, index = indexes).join(sex_and_eid_columns)
    #
    #     return df_rescaled
    #
    #
    # def inverse_normalise_dataset(self, df_rescaled):
    #     if self.target == 'Sex':
    #         return df_rescaled
    #     elif self.target == 'Age':
    #         df_noscaled = df_rescaled
    #         if hasattr(self, 'scaler'):
    #             df_noscaled['pred'] = self.scaler.inverse_transform(df_noscaled['pred'].values.reshape(-1, 1))
    #             #df_noscaled['real'] = self.scaler.inverse_transform(df_noscaled['real'].values.reshape(-1, 1))
    #         return df_noscaled
    #     else :
    #         raise ValueError('dataframe is not rescaled')


    def save_features(self, cols):
        if 'Cluster' in self.dataset:
            dataset_proper = self.dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
        else :
            dataset_proper = self.dataset
        if not hasattr(self, 'features_imp') and self.model_name != 'Correlation' :
            raise ValueError('Features importance not trained')
        save_features_to_csv(cols, self.features_imp, self.target, dataset_proper, self.model_name, sd = False)
        save_features_to_csv(cols, self.features_imp_sd, self.target, dataset_proper, self.model_name, sd = True)


    def save_predictions(self, predicts_df, step):
        if 'Cluster' in self.dataset:
            dataset_proper = self.dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
        else :
            dataset_proper = self.dataset
        if not hasattr(self, 'best_params'):
            raise ValueError('Predictions not trained')
        save_predictions_to_csv(predicts_df, step, self.target, dataset_proper, self.model_name, self.fold, self.best_params)
