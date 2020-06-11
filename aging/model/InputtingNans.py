import glob
import pandas as pd
from sklearn.linear_model import LinearRegression

cols_ethnicity_full = ['Do_not_know', 'Prefer_not_to_answer', 'NA', 'White', 'British',
       'Irish', 'White_Other', 'Mixed', 'White_and_Black_Caribbean',
       'White_and_Black_African', 'White_and_Asian', 'Mixed_Other', 'Asian',
       'Indian', 'Pakistani', 'Bangladeshi', 'Asian_Other', 'Black',
       'Caribbean', 'African', 'Black_Other', 'Chinese', 'Other_ethnicity',
       'Other']

def load_raw_data(path_raw,
                  path_output,
                  path_inputs,
                  path_ethnicities = '/n/groups/patel/samuel/ethnicities.csv'):
    final_df = pd.read_csv(path_raw).set_index('id')
    df_ethnicity = df_ethnicity = pd.read_csv(path_ethnicities).set_index('eid')
    cols_ethnicity = ['White', 'Mixed', 'Black', 'Asian', 'Other', 'Chinese']
    df_ethnicity = pd.DataFrame(df_ethnicity[cols_ethnicity].idxmax(axis = 1))
    df_ethnicity.columns = ['Ethnicity']
    final_df = final_df.reset_index().merge(df_ethnicity, on ='eid').set_index('id')

    features = [ elem for elem in final_df.columns if elem not in cols_ethnicity_full + ['Sex', 'Age when attended assessment centre', 'Ethnicity', 'eid']]
    return features, final_df

# def load_raw_data(path_raw,
#                   path_output,
#                   path_inputs,
#                   path_ethnicities = '/n/groups/patel/samuel/ethnicities.csv'):
#     final_df = pd.read_csv(path_raw).set_index('id')
#     df_ethnicity = df_ethnicity = pd.read_csv(path_ethnicities).set_index('eid')
#     cols_ethnicity = ['White', 'Mixed', 'Black', 'Asian', 'Other', 'Chinese']
#     df_ethnicity = pd.DataFrame(df_ethnicity[cols_ethnicity].idxmax(axis = 1))
#     df_ethnicity.columns = ['Ethnicity']
#     final_df = final_df.reset_index().merge(df_ethnicity, on ='eid').set_index('id')
#
#     list_int_cols = []
#     for idx, elem in enumerate(glob.glob(path_inputs + '*.csv')):
#         print(elem)
#         df_temp = pd.read_csv(elem, nrows = 1).set_index('id')
#         used_cols = [elem for elem in df_temp.columns if elem not in ['Sex', 'eid', 'Age when attended assessment centre'] + cols_ethnicity]
#         int_cols = list(df_temp[used_cols].select_dtypes(int).columns)
#         list_int_cols += int_cols
#
#     ethnicity_features = df_ethnicity.columns
#     continuous_cols = [elem for elem in final_df.columns if elem not in list_int_cols and elem not in ethnicity_features and elem not in ['Sex', 'Age when attented assessment centre', 'eid']]
#
#     return list_int_cols, continuous_cols, final_df



def compute_linear_coefficients_for_each_col(final_df, col):
    age_sex_ethnicity_features = ['Sex', 'Age when attended assessment centre', 'Ethnicity']
    coefs_col = pd.DataFrame(columns= [col, 'Sex', 'Ethnicity'])
    column = final_df[[col, 'eid']  + age_sex_ethnicity_features].iloc[:50000]
    distinct_eid_col = column.eid.drop_duplicates().values
    is_longitudinal = (column.groupby('eid').count() > 1).any().any()

    if not is_longitudinal:
        raise ValueError('Feature is not longitudinal')
    else :
        ## Create weights by sex and ethnicty
        for eid in distinct_eid_col:
            points = column[column.eid == eid].dropna()
            num_points = (~points[col].isna()).sum()
            if num_points == 1 or num_points == 0:
                continue
            else :
                if num_points == 2:
                    point1 = points.iloc[0]
                    point2 = points.iloc[1]
                    coef = (point2[col] - point1[col])/(point2['Age when attended assessment centre'] - point1['Age when attented assessment centre'])

                elif num_points in [3, 4]:
                    y = points[col].values.reshape(-1, 1)
                    x = points['Age when attented assessment centre'].values.reshape(-1, 1)
                    lin  = LinearRegression()
                    lin.fit(x, y)
                    coef = lin.coef_[0][0]
                else :
                    raise ValueError('not the right number of points')
                coefs_col = coefs_col.append(pd.Series([coef, points['Sex'].mean(), points['Ethnicity'].min()], index=[col, 'Sex', 'Ethnicity'], name= eid))
        coefs_mean = coefs_col.groupby(['Sex', 'Ethnicity']).mean()
        return coefs_mean, distinct_eid_col, column



def input_variables_in_column(col, column, distinct_eid_col, coefs_mean):
    categorical = (column[col].max() == 1) and (column[col].min() == 0)
    def recenter_between_0_1(value_):
        if value_ < 0:
            return 0
        elif value_ > 1:
            return 1
        else :
            return value_

    for eid in distinct_eid_col:
        points = column[column.eid == eid]

        ## inputting or not :
        if points[col].isna().any():

            ## count number of availaible points:
            num_avail = points.shape[0]
            num_avail_filled = len(points[col].dropna())
            if num_avail == 1 or num_avail_filled == 0:
                continue
            elif num_avail == 2:
                missing_point = points[points[col].isna()].iloc[0]
                valid_point = points[~points[col].isna()].iloc[0]
                ethnicity = valid_point['Ethnicity']
                sex = valid_point['Sex']
                age_missing = missing_point['Age when attented assessment centre']
                age_valid = valid_point['Age when attented assessment centre']
                valid_value = valid_point[col]
                coef_ = coefs_mean.loc[sex, ethnicity].values
                missing_value = valid_value + (age_missing - age_valid) * coef_
                if categorical:
                    missing_value = recenter_between_0_1(missing_value)
                column.loc[missing_point.name, col] = missing_value

            elif num_avail == 3:
                if num_avail_filled == 2:
                    missing_point = points[points[col].isna()].iloc[0]
                    valid_point_1 = points[~points[col].isna()].iloc[0]
                    valid_point_2 = points[~points[col].isna()].iloc[1]
                    ethnicity = missing_point['Ethnicity']
                    sex = missing_point['Sex']
                    coef_ = coefs_mean.loc[sex, ethnicity].values
                    age_missing = missing_point['Age when attented assessment centre']

                    age_valid_1 = valid_point_1['Age when attented assessment centre']
                    age_valid_2 = valid_point_2['Age when attented assessment centre']

                    estimated_1 = valid_point_1[col] + (age_missing - age_valid_1) * coef_
                    estimated_2 = valid_point_2[col] + (age_missing - age_valid_2) * coef_

                    dist_1 = abs(age_valid_1 - age_missing)
                    dist_2 = abs(age_valid_2 - age_missing)

                    missing_value = (estimated_1/dist_1 + estimated_2/dist_2) / (1/dist_1 + 1/dist_2)
                    if categorical:
                        missing_value = recenter_between_0_1(missing_value)
                    column.loc[missing_point.name, col] = missing_value
                else : # 2 missing points :
                    missing_point_1 = points[points[col].isna()].iloc[0]
                    missing_point_2 = points[points[col].isna()].iloc[1]
                    valid_point = points[~points[col].isna()].iloc[0]
                    ethnicity = valid_point['Ethnicity']
                    sex = valid_point['Sex']
                    age_missing_1 = missing_point_1['Age when attented assessment centre']
                    age_missing_2 = missing_point_2['Age when attented assessment centre']
                    age_valid = valid_point['Age when attented assessment centre']
                    valid_value = valid_point[col]
                    coef_ = coefs_mean.loc[sex, ethnicity].values
                    missing_value_1 = valid_value + (age_missing_1 - age_valid) * coef_
                    missing_value_2 = valid_value + (age_missing_2 - age_valid) * coef_
                    if categorical:
                        missing_value_1 = recenter_between_0_1(missing_value_1)
                        missing_value_2 = recenter_between_0_1(missing_value_2)
                    column.loc[missing_point_1.name, col] = missing_value_1
                    column.loc[missing_point_2.name, col] = missing_value_2
            elif num_avail == 4:
                if num_avail_filled == 1:
                    missing_point_1 = points[points[col].isna()].iloc[0]
                    missing_point_2 = points[points[col].isna()].iloc[1]
                    missing_point_3 = points[points[col].isna()].iloc[2]
                    valid_point = points[~points[col].isna()].iloc[0]
                    ethnicity = valid_point['Ethnicity']
                    sex = valid_point['Sex']
                    age_missing_1 = missing_point_1['Age when attented assessment centre']
                    age_missing_2 = missing_point_2['Age when attented assessment centre']
                    age_missing_3 = missing_point_3['Age when attented assessment centre']
                    age_valid = valid_point['Age when attented assessment centre']
                    valid_value = valid_point[col]
                    coef_ = coefs_mean.loc[sex, ethnicity].values
                    missing_value_1 = valid_value + (age_missing_1 - age_valid) * coef_
                    missing_value_2 = valid_value + (age_missing_2 - age_valid) * coef_
                    missing_value_3 = valid_value + (age_missing_4 - age_valid) * coef_
                    if categorical :
                        missing_value_1 = recenter_between_0_1(missing_value_1)
                        missing_value_2 = recenter_between_0_1(missing_value_2)
                        missing_value_3 = recenter_between_0_1(missing_value_3)


                    column.loc[missing_point_1.name, col] = missing_value_1
                    column.loc[missing_point_2.name, col] = missing_value_2
                    column.loc[missing_point_3.name, col] = missing_value_3

                elif num_avail_filled == 2:
                    missing_point_1 = points[points[col].isna()].iloc[0]
                    missing_point_2 = points[points[col].isna()].iloc[1]
                    valid_point_1 = points[~points[col].isna()].iloc[0]
                    valid_point_2 = points[~points[col].isna()].iloc[1]

                    ethnicity = missing_point_1['Ethnicity']
                    sex = missing_point_1['Sex']
                    coef_ = coefs_mean.loc[sex, ethnicity].values

                    age_missing_1 = missing_point_1['Age when attented assessment centre']
                    age_missing_2 = missing_point_2['Age when attented assessment centre']

                    age_valid_1 = valid_point_1['Age when attented assessment centre']
                    age_valid_2 = valid_point_2['Age when attented assessment centre']

                    estimated_1_missing_1 = valid_point_1[col] + (age_missing_1 - age_valid_1) * coef_
                    estimated_2_missing_1 = valid_point_2[col] + (age_missing_1 - age_valid_2) * coef_
                    estimated_1_missing_2 = valid_point_1[col] + (age_missing_2 - age_valid_1) * coef_
                    estimated_2_missing_2 = valid_point_2[col] + (age_missing_2 - age_valid_2) * coef_

                    dist_1_missing_1 = abs(age_valid_1 - age_missing_1)
                    dist_2_missing_1 = abs(age_valid_2 - age_missing_1)

                    dist_1_missing_2 = abs(age_valid_1 - age_missing_2)
                    dist_2_missing_2 = abs(age_valid_2 - age_missing_2)

                    missing_value_1 = (estimated_1_missing_1/dist_1_missing_1 + estimated_2_missing_1/dist_2_missing_1) / (1/dist_1_missing_1 + 1/dist_2_missing_1)
                    missing_value_2 = (estimated_1_missing_2/dist_1_missing_2 + estimated_2_missing_2/dist_2_missing_2) / (1/dist_1_missing_2 + 1/dist_2_missing_2)

                    if categorical:
                        missing_value_1 = recenter_between_0_1(missing_value_1)
                        missing_value_2 = recenter_between_0_1(missing_value_2)
                    column.loc[missing_point_1.name, col] = missing_value_1
                    column.loc[missing_point_2.name, col] = missing_value_2


                elif num_avail_filled == 3:
                    missing_point = points[points[col].isna()].iloc[0]
                    valid_point_1 = points[~points[col].isna()].iloc[0]
                    valid_point_2 = points[~points[col].isna()].iloc[1]
                    valid_point_3 = points[~points[col].isna()].iloc[2]
                    ethnicity = missing_point['Ethnicity']
                    sex = missing_point['Sex']
                    coef_ = coefs_mean.loc[sex, ethnicity].values
                    age_missing = missing_point['Age when attented assessment centre']

                    age_valid_1 = valid_point_1['Age when attented assessment centre']
                    age_valid_2 = valid_point_2['Age when attented assessment centre']
                    age_valid_3 = valid_point_3['Age when attented assessment centre']

                    estimated_1 = valid_point_1[col] + (age_missing - age_valid_1) * coef_
                    estimated_2 = valid_point_2[col] + (age_missing - age_valid_2) * coef_
                    estimated_3 = valid_point_3[col] + (age_missing - age_valid_3) * coef_

                    dist_1 = abs(age_valid_1 - age_missing)
                    dist_2 = abs(age_valid_2 - age_missing)
                    dist_3 = abs(age_valid_3 - age_missing)
                    missing_value = (estimated_1/dist_1 + estimated_2/dist_2 + estimated_3/dist_3) / (1/dist_1 + 1/dist_2 + 1/dist_3)
                    if categorical:
                        missing_value = recenter_between_0_1(missing_value)
                    column.loc[missing_point.name, col] = missing_value
    return column

def compute_coefs_and_input(final_df, col):
    print("Compute mean of coef : %s" % col)
    coefs_mean, distinct_eid_col, column = compute_linear_coefficients_for_each_col(final_df, col)
    print("Done , input missing data in %s" % col )
    column_modified = input_variables_in_column(col, column, distinct_eid_col, coefs_mean)
    return column_modified
