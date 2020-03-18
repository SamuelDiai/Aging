
import sys
import os
if sys.platform == 'linux':
	sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
	sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.general_predictor import GeneralPredictor


name = sys.argv[1]
n_iter = int(sys.argv[2])
target = sys.argv[3]
dataset = sys.argv[4]
n_splits = int(sys.argv[5])



hyperparameters = dict()
hyperparameters['name'] = name
hyperparameters['n_splits'] = n_splits
hyperparameters['n_iter'] = n_iter
hyperparameters['target'] = target
hyperparameters['dataset'] = dataset
print(hyperparameters)

gp = GeneralPredictor(name, -1, n_splits, n_iter, target, dataset)
print("Loading Dataset")
df = gp.load_dataset()
print("Dataset Loaded, optimizing hyper")
df_scaled = gp.normalise_dataset(df)
feature_importance_ = gp.feature_importance(df_scaled, n_iter, n_splits)
print("Feature importance over, saving file")
print("task complete")
