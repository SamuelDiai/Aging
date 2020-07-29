import numpy as np
import sys
import os
import glob
import pandas as pd

if sys.platform == 'linux':
	sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
	sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_data import load_data, dict_dataset_to_organ_and_view
from aging.model.load_and_save_survival_data import path_predictions_survival
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



if 'Cluster' in dataset:
	dataset_proper = dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
	organ = 'Cluster'
	view = 'main'
else :
	dataset_proper = dataset
	organ, view, transformation =  dict_dataset_to_organ_and_view[dataset_proper]

list_files = glob.glob(path_predictions_survival + '/*%s_%s_%s*%s*%s*.csv' % (organ, view, transformation, target, model))

list_train = [elem for elem in list_files if 'train.csv' in elem]
list_test = [elem for elem in list_files if 'test.csv' in elem]
list_val = [elem for elem in list_files if 'val.csv' in elem]

if len(list_train) == outer_splits and len(list_test) == outer_splits and len(list_val) == outer_splits :

	df_train = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_train])
	df_test = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_test])
	df_val = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_val])

	if 'chemestry' in organ:
		organ = organ.replace('chemestry', 'chemistry')


	#print('/n/groups/patel/samuel/preds_alan/Predictions_instances_%s_%s_%s_raw_%s_0_0_0_0_0_0_0_train.csv' % ( target, organ, view, model))
	df_train[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/Survival/preds/PredictionsSurvival_instances_%s_%s_%s_%s_raw_%s_0_0_0_0_0_0_0_train.csv' % ( target, organ, view, transformation, model))
	df_test[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/Survival/preds/PredictionsSurvival_instances_%s_%s_%s_%s_raw_%s_0_0_0_0_0_0_0_test.csv' % ( target, organ, view, transformation, model))
	df_val[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/Survival/preds/PredictionsSurvival_instances_%s_%s_%s_%s_raw_%s_0_0_0_0_0_0_0_val.csv' % ( target, organ, view, transformation, model))


else :
	raise ValueError("ONE OF THE OUTER JOB HAS FAILED ! ")
