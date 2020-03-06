#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:17:03 2020

@author: grael
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_roc_curve, make_scorer
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from skimage.measure import regionprops_table



def spearman_score(true_probs, pred_probs):
    """
    A metric to assess multilabel regression performance using the 
    un-weighted class-average 0-clipped Spearman Correlation.
    
    can be passes to sklearn make_scorer
    """
    
    m, c = true_probs.shape
    
    assert m == pred_probs.shape[0]
    
    if true_probs.ndim == 1:
        s, p = spearmanr(true_probs, pred_probs)
    elif true_probs.ndim == 2:
        assert c == pred_probs.shape[1]
        
        s_list = []
        for n in range(c):
            s, p = spearmanr(true_probs[:,n], pred_probs[:,n])
            s_list.append(s)
        s = np.mean(np.array(s_list))
        
    else:
        raise ValueError
    
    # don't allow negative correlations to be rewarded.
    s = np.clip(s, 0., 1.)
    return s

scorer = make_scorer(spearman_score, needs_probab=True)


"""ROC Plot function"""

def roc_plot(fitted_model, test_x, test_y, train_x, train_y, feature_problem='Morpho-Only'):
    f, ax = plt.subplots(1,1, dpi=200, figsize=(6,6))
    plot_roc_curve(fitted_model, test_x, test_y, ax=ax, label=' '.join([feature_problem, 'Test']))
    plot_roc_curve(fitted_model, train_x, train_y, ax=ax, label=' '.join([feature_problem, 'Train']), linestyle='--')
    plt.show()

    
"""
## An AUC estimate that doesn't require explicit construction of an ROC curve
## Source: Hand & Till (2001)
auc <- function( probs, preds, ref )
{
    stopifnot( length(probs) == length(preds) )
    jp <- which(preds==ref); np <- length(jp)
    jn <- which(preds!=ref); nn <- length(jn)
    s0 <- sum( rank(probs)[jp] )
    (s0 - np*(np+1) / 2) / np / nn
}
"""


def auc(probs, preds, ref):
    """
    probs: real valued vector (true label probabilities)
    
    preds: multi-class binary prediction (predicted labels)
    
    ref: the indicator of one of the labels.
    """
    assert type(probs) is np.ndarray
    assert type(preds) is np.ndarray
    assert probs.shape[0] == preds.shape[0]
    
    jp = np.flatnonzero(preds==ref)
    num_p = jp.shape[0]
    jn = np.flatnonzero(preds!=ref)
    num_n = jn.shape[0]
    s0 = np.sum(np.argsort(probs)[jp])
    return (s0 - num_p*(num_p+1) / 2) / num_p / num_n
    


"""Load lung1 and Images; extract new features.
"""

root = '/Users/grael/hackathon_data'

lung1 = pd.read_csv(os.path.join(root, 'Lung1.csv'), index_col=0)
lung2 = pd.read_csv(os.path.join(root, 'Lung2.csv'), index_col=0)
lung3 = pd.read_csv(os.path.join(root, 'Lung3.csv'), index_col=0)





import tifffile


# DAPI 4 is channel 12 (0-indexed).
im_name1 = os.path.join(root, 'LUNG-1-LN_40X.ome.tif')
imsave_name1 = os.path.join(root, 'LUNG-1-LN_40X_DAPI1.tif')
if not os.path.isfile(imsave_name1):
    dapi_lung1 = np.squeeze(tifffile.imread(im_name1)[4, :, :])
    tifffile.imsave(imsave_name1, dapi_lung1)
else:
    dapi_lung1 = tifffile.imread(imsave_name1)


im_name2 = os.path.join(root, 'LUNG-2-BR_40X.ome.tif')
imsave_name2 = os.path.join(root, 'LUNG-2-BR_40X_DAPI1.tif')
if not os.path.isfile(imsave_name2):
    dapi_lung2 = np.squeeze(tifffile.imread(im_name2)[4, :, :])
    tifffile.imsave(imsave_name2, dapi_lung2)
else:
    dapi_lung2 = tifffile.imread(imsave_name2)
    
im_name3 = os.path.join(root, 'LUNG-3-PR_40X.ome.tif')
imsave_name3 = os.path.join(root, 'LUNG-3-PR_40X_DAPI1.tif')
if not os.path.isfile(imsave_name3):
    dapi_lung3 = np.squeeze(tifffile.imread(im_name3)[4, :, :])
    tifffile.imsave(imsave_name3, dapi_lung3)
else:
    dapi_lung3 = tifffile.imread(imsave_name3)
    
# don't extract features from segmentation basins with ID over 87,500 in Lung1

mask_lung1 = tifffile.imread(os.path.join(root, 'LUNG-1-LN_40X_Seg_labeled.tif'))
mask_lung2 = tifffile.imread(os.path.join(root, 'LUNG-2-BR_40X_Seg_labeled.tif'))
mask_lung3 = tifffile.imread(os.path.join(root, 'LUNG-3-PR_40X_Seg_labeled.tif'))

# we needed to clean up the images so they were the same shape.
mask_lung1 = mask_lung1[:, :9666]
dapi_lung3 = dapi_lung3[:, :14447]

properties = [
        'label',
        'area',
        'perimeter',
        'eccentricity',
        'extent',
        'mean_intensity',
        # 'weighted_moments_hu',
        # 'weighted_moments_central',
        # 'weighted_moments_normalized'
        ]


# dapi_lung1 = img_as_float(dapi_lung1[:1000, : 1000])
# mask_lung1 = mask_lung1[:1000, : 1000]

# dapi_lung2 = img_as_float(dapi_lung2[:1000, : 1000])
# mask_lung2 = mask_lung2[:1000, : 1000]

# dapi_lung3 = img_as_float(dapi_lung3[:1000, : 1000])
# mask_lung3 = mask_lung3[:1000, : 1000]


rp1 = regionprops_table(
        mask_lung1,
        intensity_image=dapi_lung1,
        properties=properties)

rp2 = regionprops_table(
        mask_lung2,
        intensity_image=dapi_lung2,
        properties=properties)

rp3 = regionprops_table(
        mask_lung3,
        intensity_image=dapi_lung3,
        properties=properties)


rp1_df = pd.DataFrame(rp1, index=rp1['label']).drop(columns=['label'])
rp2_df = pd.DataFrame(rp2, index=rp2['label']).drop(columns=['label'])
rp3_df = pd.DataFrame(rp3, index=rp3['label']).drop(columns=['label'])


properties2 = [
        'label',
        'intensity_image',
        'image',
        'coords',
        'bbox']

rpi1 = regionprops_table(
        mask_lung1,
        intensity_image=dapi_lung1,
        properties=properties2)

rpi2 = regionprops_table(
        mask_lung2,
        intensity_image=dapi_lung2,
        properties=properties2)

rpi3 = regionprops_table(
        mask_lung3,
        intensity_image=dapi_lung3,
        properties=properties2)


rpi1_df = pd.DataFrame(rpi1, index=rpi1['label']).drop(columns=['label'])
rpi2_df = pd.DataFrame(rpi2, index=rpi2['label']).drop(columns=['label'])
rpi3_df = pd.DataFrame(rpi3, index=rpi3['label']).drop(columns=['label'])



"""
Regionprops bounding box format:
Bounding box (min_row, min_col, max_row, max_col)
"""
def ints_std(df):
    out = []
    for i in range(df.shape[0]):
        intensity_image, mask_image, coords, min_row, min_col, max_row, max_col = df.iloc[i, :]
        out.append(np.std(intensity_image[mask_image]))
    return out


rp1_df['std_intensity'] = ints_std(rpi1_df)
rp2_df['std_intensity'] = ints_std(rpi2_df)
rp3_df['std_intensity'] = ints_std(rpi3_df)


# we found an artifact in the bottom right corner of the segmentation mask.
rp1_df = rp1_df.iloc[:87500, :]
lung1 = lung1.iloc[:87500, :]


rp1_df.to_csv(os.path.join(root, 'Lung1_new_features.csv'))
rp2_df.to_csv(os.path.join(root, 'Lung2_new_features.csv'))
rp3_df.to_csv(os.path.join(root, 'Lung3_new_features.csv'))



lung1['mean_intensity'] = rp1_df['mean_intensity']
lung2['mean_intensity'] = rp2_df['mean_intensity']
lung3['mean_intensity'] = rp3_df['mean_intensity']
lung1['std_intensity'] = rp1_df['std_intensity']
lung2['std_intensity'] = rp2_df['std_intensity']
lung3['std_intensity'] = rp3_df['std_intensity']



# Testing for Calum's code.
Label2Group = {'Immune':0, 'Stroma':1, 'Tumor':2}
Group2Label = {0:'Immune', 1:'Stroma', 2:'Tumor'}

morphological_features = ['Area',
                        'Eccentricity',
                        'Solidity',
                        'Extent',
                        'EulerNumber',
                        'Perimeter',
                        'MajorAxisLength',
                        'MinorAxisLength',
                        'Orientation',
                        'X_position',
                        'Y_position',
                        'mean_intensity',
                        'std_intensity']

data_fil = lung1.loc[(lung1.loc[:, ('Neighbor_1', 'Neighbor_2',
    'Neighbor_3', 'Neighbor_4', 'Neighbor_5')] != 0).all(axis=1) & (lung1['mean_intensity'] != 0), :]

Neighbour1 = lung1.loc[lung1.loc[:, 'Neighbor_1'], morphological_features]
Neighbor_1 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_2 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_3 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_4 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_5 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
for i in Neighbor_1.index:
    Neighbor_1.loc[i, :] = lung1.loc[lung1.loc[i, 'Neighbor_1'], morphological_features]
    Neighbor_2.loc[i, :] = lung1.loc[lung1.loc[i, 'Neighbor_2'], morphological_features]
    Neighbor_3.loc[i, :] = lung1.loc[lung1.loc[i, 'Neighbor_3'], morphological_features]
    Neighbor_4.loc[i, :] = lung1.loc[lung1.loc[i, 'Neighbor_4'], morphological_features]
    Neighbor_5.loc[i, :] = lung1.loc[lung1.loc[i, 'Neighbor_5'], morphological_features]
Neighbor_1.columns = ['{}_{}'.format(col, 1) for col in Neighbor_1.columns]
Neighbor_2.columns = ['{}_{}'.format(col, 2) for col in Neighbor_2.columns]
Neighbor_3.columns = ['{}_{}'.format(col, 3) for col in Neighbor_3.columns]
Neighbor_4.columns = ['{}_{}'.format(col, 4) for col in Neighbor_4.columns]
Neighbor_5.columns = ['{}_{}'.format(col, 5) for col in Neighbor_5.columns]
Neighbors = Neighbor_1.join(Neighbor_2).join(Neighbor_3).join(Neighbor_4).join(Neighbor_5)