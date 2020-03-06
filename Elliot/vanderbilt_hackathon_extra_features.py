#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:33:59 2020

@author: grael
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os


root = '/Users/grael/hackathon_data'

nbr1 = pd.read_csv(os.path.join(root, 'Neighbors_new_features_1.csv'), index_col=0)
nbr2 = pd.read_csv(os.path.join(root, 'Neighbors_new_features_2.csv'), index_col=0)
nbr3 = pd.read_csv(os.path.join(root, 'Neighbors_new_features_3.csv'), index_col=0)



plt.figure()
sns.clustermap(nbr1.corr(), center=0)
plt.show()


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

short_features = [c for c in nbr1.columns if ((
    ('Area' in c) or ('mean_' in c) or ('std' in c)
    or ('AxisLength' in c) or ('Perimeter' in c) or
    ('Eccentricity' in c) or ('Extent' in c)) and ('_1' in c))]

plt.figure()
sns.clustermap(nbr1[short_features].corr(), center=0)
plt.show()



# orientation_features = [c for c in nbr1.columns if ('Orientation' in c)]

# odf1 = nbr1[orientation_features].T.corr().T
# odf2 = nbr2[orientation_features].T.corr().T
# odf3 = nbr3[orientation_features].T.corr().T


# nbr1['OrientationCorrelation'] = odf1
# nbr2['OrientationCorrelation'] = odf2
# nbr3['OrientationCorrelation'] = odf3


save_features = [c for c in nbr1.columns if ((
    ('Area' in c) or ('mean_' in c) or ('std' in c)
    or ('AxisLength' in c) or ('Perimeter' in c) or
    ('Eccentricity' in c) or ('Extent' in c) or ('Solidity' in c))
    # or ('OrientationCorrelation' in c)
    )]

nbr1[save_features].to_csv(os.path.join(root, 'Lung1_neighborhood_features.csv'), header=True)
nbr2[save_features].to_csv(os.path.join(root, 'Lung2_neighborhood_features.csv'), header=True)
nbr3[save_features].to_csv(os.path.join(root, 'Lung3_neighborhood_features.csv'), header=True)