
from .general_predictor import *
from .load_and_save_data import load_data, save_features_to_csv, save_predictions_to_csv
from ..processing.abdominal_composition_processing import read_abdominal_data
from ..processing.brain_processing import read_grey_matter_volumes_data, read_subcortical_volumes_data, read_brain_data
from ..processing.heart_processing import read_heart_data, read_heart_size_data, read_heart_PWA_data
from ..processing.body_composition_processing import read_body_composition_data
from ..processing.bone_composition_processing import read_bone_composition_data

class GeneralPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, target, dataset, fold):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter)
        self.fold = fold
        self.dataset = dataset
        if target == 'Sex':
            self.scoring = 'f1'
            self.target = 'Sex'
            if model == 'ElasticNet':
                self.model = ElasticNet(max_iter = 2000)
            elif model == 'RandomForest':
                self.model = RandomForestClassifier()
            elif model == 'GradientBoosting':
                self.model = GradientBoostingClassifier()
            elif model == 'Xgboost':
                self.model = XGBClassifier()
            elif model == 'LightGbm':
                self.model = LGBMClassifier()
            elif model == 'NeuralNetwork':
                self.model = MLPClassifier(solver = 'adam')
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
                self.model = MLPRegressor(solver = 'adam')
        else :
            raise ValueError('target : "%s" not valid, please enter "Sex" or "Age"' % target)

    def feature_importance(self, df):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
            y = df['Sex'].values
            self.features_importance_(X, y, self.scoring)
            return df.drop(columns = ['Sex', 'Age when attended assessment centre']).columns
        elif self.target == 'Age':
            X = df.drop(columns = ['Age when attended assessment centre']).values
            y = df['Age when attended assessment centre'].values
            self.features_importance_(X, y, self.scoring)
            return df.drop(columns = ['Age when attended assessment centre']).columns
        else :
            raise ValueError('GeneralPredictor not instancied')





    def optimize_hyperparameters_fold(self, df):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
            y = df['Sex'].values
        elif self.target == 'Age':
            X = df.drop(columns = ['Age when attended assessment centre']).values
            y = df['Age when attended assessment centre'].values
        else :
            raise ValueError('GeneralPredictor not instancied')
        return self.optimize_hyperparameters_fold_(X, y, df.index, self.scoring, self.fold)

    def save_features(self, cols):
        if not hasattr(self, 'features_imp'):
            raise ValueError('Features importance not trained')
        save_features_to_csv(cols, self.features_imp, self.target, self.dataset, self.model_name)

    def load_dataset(self, **kwargs):
        return load_data(self.dataset, **kwargs)

    def save_predictions(self, predicts_df, step):
        if not hasattr(self, 'best_params'):
            raise ValueError('Predictions not trained')
        save_predictions_to_csv(predicts_df, step, self.target, self.dataset, self.model_name, self.fold, self.best_params)


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
