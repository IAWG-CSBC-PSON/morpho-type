import pandas as pd
import numpy as np
import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

from train_utils import my_GridSearchCV
sys.path.append('/cellar/users/y8qin/Data/my_utils/')
from file_utils import *

parser = argparse.ArgumentParser(description='Train using morphological features with Random Forest Classifier.')
parser.add_argument('--testSample', type=int, help='Sample number used for testing.')
args = parser.parse_args()

testSample = args.testSample
trainSample = [1,2,3]
trainSample.remove(testSample)

lung = load_obj('/cellar/users/y8qin/Data/image_hackathon/morpho-type/Yue/lung.pkl')
train_lung = lung[lung['sample'].isin(trainSample)]
test_lung = lung[lung['sample'] == testSample]

morpho_feat = ['Area', 
               'Eccentricity', 
               'Solidity', 
               'Extent',  
               'Perimeter', 
               'MajorAxisLength', 
               'MinorAxisLength']


train_X = train_lung[morpho_feat].values
test_X = test_lung[morpho_feat].values

label_map = {'Stroma':0, 'Immune':1, 'Tumor':2}
label_idx_map = {0:'Stroma', 1:'Immune', 2:'Tumor'}

train_y = train_lung['Label'].map(label_map).values
test_y = test_lung['Label'].map(label_map).values

# Random Forest
rf_kwargs = {
    'n_estimators': 200,
    'class_weight': 'balanced', 
    'n_jobs':8}
rf_tuning = {
    'min_samples_split': [.05, .1, .2],
    'max_features': ['sqrt', .5, .75]}
rf = Pipeline(
    [('classification', my_GridSearchCV(
         RandomForestClassifier,
         rf_kwargs, rf_tuning, scoring='f1_macro'))])
rf.fit(train_X, train_y)
pred = rf.predict(test_X)
pred_proba = rf.predict_proba(test_X)
print('Test data accuracy: {}'.format(accuracy_score(test_y, pred)))

result = pd.DataFrame(pred_proba, columns=['Stroma_proba', 'Immune_proba', 'Tumor_proba'])
result['CellID'] = test_lung['CellID'].values
result['Pred'] = [label_idx_map[x] for x in pred]
column_order = ['CellID', 'Pred', 'Stroma_proba', 'Immune_proba', 'Tumor_proba']
result = result[column_order]

outprefix = '/cellar/users/y8qin/Data/image_hackathon/morpho-type/Yue/output/rf_classifier_testSample_{}'.format(testSample)
# Ouput prediction file for performance
result.to_csv('{}.pred.csv'.format(outprefix), columns=['CellID', 'Pred'], index=False)
# Ouput all result
result.to_csv('{}.pred_proba.csv'.format(outprefix), index=False)
# Save trained model
save_obj(rf, '{}.model.pkl'.format(outprefix))