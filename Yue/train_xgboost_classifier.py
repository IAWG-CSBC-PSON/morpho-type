import pandas as pd
import numpy as np
import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from train_utils import my_GridSearchCV
from plot_utils import plot_roc
sys.path.append('/cellar/users/y8qin/Data/my_utils/')
from file_utils import *

parser = argparse.ArgumentParser(description='Train using morphological features with XGBoost.')
parser.add_argument('--testSample', type=int, help='Sample number used for testing.')
parser.add_argument('--morpho', action='store_true', help='Use morphological features')
parser.add_argument('--dapi', action='store_true', help='Use Dapi features')
parser.add_argument('--neighbor', action='store_true', help='Use neighbor cell features')
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
dapi_feat = ['mean_intensity', 'std_intensity']
neighbor_feat = list(np.load('/cellar/users/y8qin/Data/image_hackathon/morpho-type/Yue/neighbor_feature_list.npy'))

feat = []
feat_name = ''
if args.morpho:
    feat += morpho_feat
    feat_name += 'morpho'
if args.dapi:
    feat += dapi_feat
    feat_name += '_dapi'
if args.neighbor:
    feat += neighbor_feat
    feat_name += '_neighbor'

train_X = train_lung[feat].values
test_X = test_lung[feat].values

label_map = {'Stroma':0, 'Immune':1, 'Tumor':2}
label_idx_map = {0:'Stroma', 1:'Immune', 2:'Tumor'}

train_y = train_lung['Label'].map(label_map).values
test_y = test_lung['Label'].map(label_map).values

# XGBoost
xgb_kwargs = {
    'n_estimators': 200,
    'class_weight': 'balanced',
    'nthread': 8}
xgb_tuning = {
    'max_depth': [3, 5, 8, 10],
    'colsample_bytree': [0.5, 0.7, 0.9, 1]}
xgb = Pipeline(
    [('classification', my_GridSearchCV(
          XGBClassifier,
          xgb_kwargs, xgb_tuning, scoring='f1_macro'))])
xgb.fit(train_X, train_y)
# Check XGboost output
pred = xgb.predict(test_X)
pred_proba = xgb.predict_proba(test_X)
print('Test data accuracy: {}'.format(accuracy_score(test_y, pred)))

result = pd.DataFrame(pred_proba, columns=['Stroma_proba', 'Immune_proba', 'Tumor_proba'])
result['CellID'] = test_lung['CellID'].values
result['Pred'] = [label_idx_map[x] for x in pred]
column_order = ['CellID', 'Pred', 'Stroma_proba', 'Immune_proba', 'Tumor_proba']
result = result[column_order]

outprefix = '/cellar/users/y8qin/Data/image_hackathon/morpho-type/Yue/output/xgboost_Lung{}_{}'.format(testSample, 
                                                                                                       feat_name)
# Ouput prediction file for performance
result.to_csv('{}.pred.csv'.format(outprefix), columns=['CellID', 'Pred'], index=False)
# Ouput all result
result.to_csv('{}.pred_proba.csv'.format(outprefix), index=False)
# Save trained model
save_obj(xgb, '{}.model.pkl'.format(outprefix))
# Plot ROC curve
plot_roc(test_y, pred_proba, 'xgboost_Lung{}_{}'.format(testSample, feat_name), outfname='{}.png'.format(outprefix))

print('=== finished! ===')