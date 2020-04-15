from .general_predictor import *
from .load_and_save_environment_data import load_data as load_data, save_features_to_csv, save_predictions_to_csv



class EnvironmentPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, target_dataset, env_dataset, fold):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter)
        self.fold = fold
        self.env_dataset = env_dataset
        self.target_dataset = target_dataset
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
            self.model = MLPRegressor(solver = 'adam')
        else :
            raise ValueError('model : "%s" not valid' % model)


    def load_dataset(self, **kwargs):
        return load_data(self.env_dataset, self.target_dataset, **kwargs)


    def optimize_hyperparameters_fold(self, df):
        X = df.drop(columns = ['Age', 'residual']).values
        y = df['residual'].values
        return self.optimize_hyperparameters_fold_(X, y, df.index, 'r2', self.fold)


    def feature_importance(self, df):
        X = df.drop(columns = ['Age', 'residual']).values
        y = df['residual'].values
        self.features_importance_(X, y, 'r2')
        return df.drop(columns = ['Age', 'residual']).columns


    def normalise_dataset(self, df):

        scaler_residual = StandardScaler()
        scaler_residual.fit(df['residual'].values.reshape(-1, 1))
        self.scaler = scaler_residual

        if self.model_name == 'ElasticNet':
            cols = df.columns
            indexes = df.index
            scaler = StandardScaler()
            scaler.fit(df)
            array_rescaled = scaler.transform(df)
            return pd.DataFrame(array_rescaled, columns = cols, index = indexes)
        else :
            # Get categorical data apart from continous ones
            df_cat = df.select_dtypes(include=['int'])
            df_cont = df.drop(columns = df_cat.columns)

            cols = df_cont.columns
            indexes = df_cont.index

            # save scaler
            scaler = StandardScaler()
            scaler.fit(df_cont)

            # Scale and create Dataframe
            array_rescaled =  scaler.transform(df_cont)
            df_rescaled = pd.DataFrame(array_rescaled, columns = cols, index = indexes).join(df_cat)

            return df_rescaled

    def inverse_normalise_dataset(self, df_rescaled):
        if hasattr(self, 'scaler'):
            df_rescaled['predictions'] = self.scaler.inverse_transform(df_rescaled['predictions'].values.reshape(-1, 1))
            return df_rescaled
        else :
            raise ValueError('dataframe is not rescaled')

    def save_features(self, cols):
        if not hasattr(self, 'features_imp'):
            raise ValueError('Features importance not trained')
        save_features_to_csv(cols, self.features_imp, self.target_dataset, self.env_dataset, self.model_name)

    def save_predictions(self, predicts_df, step):
        if not hasattr(self, 'best_params'):
            raise ValueError('Predictions not trained')
        save_predictions_to_csv(predicts_df, step, self.target_dataset, self.env_dataset, self.model_name, self.fold, self.best_params)
