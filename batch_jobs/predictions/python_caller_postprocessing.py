import numpy as np
import sys
import os
import glob
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, StratifiedKFold, RandomizedSearchCV
import pandas as pd

if sys.platform == 'linux':
	sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
	sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_data import load_data
from aging.processing.base_processing import path_predictions
model = sys.argv[1]
target = sys.argv[2]
dataset = sys.argv[3]
outer_splits = int(sys.argv[4])


hyperparameters = dict()
hyperparameters['model'] = model
hyperparameters['target'] = target
hyperparameters['dataset'] = dataset
hyperparameters['outer_splits'] = outer_splits
print(hyperparameters)


# def dataset_map_fold(dataset, target, outer_splits):
#     dataset = dataset.replace('_', '')
#     df = load_data(dataset)
#     if target == 'Sex':
#         X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
#         y = df['Sex'].values
#     elif target == 'Age':
#         X = df.drop(columns = ['Age when attended assessment centre']).values
#         y = df['Age when attended assessment centre'].values
#
#     outer_cv = KFold(n_splits = outer_splits, shuffle = False, random_state = 0)
#     list_folds = [elem[1] for elem in outer_cv.split(X, y)]
#     index = df.index
#
#     index_splits = [index[list_folds[elem]].values for elem in range(outer_splits)]
#     index_split_matching = [np.array( [fold]*len(index_splits[fold])) for fold in range(outer_splits) ]
#
#     map_eid_to_fold = dict(zip(np.concatenate(index_splits), np.concatenate(index_split_matching)))
#     return map_eid_to_fold


dataset_proper = dataset.split('/')[-1].replace('.csv', '').replace('_', '.')

list_files = glob.glob( path_predictions + '*%s*%s*%s*.csv' % (target, dataset_proper, model))

list_train = [elem for elem in list_files if 'train' in elem]
list_test = [elem for elem in list_files if 'test' in elem]
list_val = [elem for elem in list_files if 'val' in elem]

if len(list_train) == outer_splits and len(list_test) == outer_splits and len(list_val) == outer_splits :

    df_train = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_train])
    df_test = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_test])
    df_val = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_val])

    # Avg df_val
    df_val = df_val.groupby('id').agg({'pred' : 'mean'})
	if 'outer_fold' not in df_val.columns :
    	df_val['outer_fold'] = np.nan

    #map_eid_to_fold = dataset_map_fold(dataset, target, outer_splits)
    #df_val['fold'] = df_val.index.map(map_eid_to_fold)

    ## Save datasets :
    #Predictions_Sex_UrineBiochemestry_100083_main_raw_GradientBoosting_0_0_0_0_test.csv
    dataset = dataset.replace('_', '')
    df_train[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/preds_final/Predictions_%s_%s_%s_main_raw_%s_0_0_0_0_train.csv' % ( target, dataset_proper, 'Cluster', model))
    df_test[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/preds_final/Predictions_%s_%s_%s_main_raw_%s_0_0_0_0_test.csv' % ( target, dataset_proper, 'Cluster', model))
    df_val[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/preds_final/Predictions_%s_%s_%s_main_raw_%s_0_0_0_0_val.csv' % ( target, dataset_proper, 'Cluster', model))

else :
    raise ValueError("ONE OF THE OUTER JOB HAS FAILED ! ")
