import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import umap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import mpl_scatter_density
from matplotlib.lines import Line2D

datafile = '/Users/gabbut02/morpho-type/data/Lung1.csv'

data = pd.read_csv(datafile, index_col='CellID')

Label2Group = {'Immune':0, 'Stroma':1, 'Tumor':2}
Group2Label = {0:'Immune', 1:'Stroma', 2:'Tumor'}

cmap = matplotlib.cm.get_cmap('viridis')

data['Group'] = [Label2Group[data.loc[i, 'Label']] for i in range(1, np.shape(data)[0]+1)]

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
                        'Y_position']

features = data[morphological_features]

data_fil = data.loc[(data.loc[:, ('Neighbor_1', 'Neighbor_2', 
    'Neighbor_3', 'Neighbor_4', 'Neighbor_5')] != 0).all(axis=1), :]   

# Neighbour1 = data.loc[data.loc[:, 'Neighbor_1'], morphological_features]
Neighbor_1 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_2 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_3 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_4 = pd.DataFrame(index=data_fil.index, columns=morphological_features)
Neighbor_5 = pd.DataFrame(index=data_fil.index, columns=morphological_features)

for i in Neighbor_1.index:
    Neighbor_1.loc[i, :] = data.loc[data.loc[i, 'Neighbor_1'], morphological_features]
    Neighbor_2.loc[i, :] = data.loc[data.loc[i, 'Neighbor_2'], morphological_features]
    Neighbor_3.loc[i, :] = data.loc[data.loc[i, 'Neighbor_3'], morphological_features]
    Neighbor_4.loc[i, :] = data.loc[data.loc[i, 'Neighbor_4'], morphological_features]
    Neighbor_5.loc[i, :] = data.loc[data.loc[i, 'Neighbor_5'], morphological_features]