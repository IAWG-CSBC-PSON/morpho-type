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
import colorsys
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
                        'Orientation']

features = data[morphological_features]

clf = RandomForestClassifier(n_estimators=100)

clf.fit(features, data['Label'])

prediction = clf.predict(features)

features_norm = np.arcsinh((features )/ np.std(features, 0))

reducer = umap.UMAP(
    n_neighbors=200,
    min_dist=0,
    n_components=2,
    random_state=42,
    metric='manhattan',
    verbose = True
)
reducer.fit(features_norm.values, y=data['Group'])

umap_results = reducer.transform(features_norm.values)

data['x_umap'] = umap_results[:,0]
data['y_umap'] = umap_results[:,1]

groups = data.groupby('Label')
# colors = get_colors( data['Label'].unique() )

fig, ax = plt.subplots()
# legend_elements = list()
scatter = ax.scatter(data.x_umap, data.y_umap, c=data.Group, s=0.1, cmap='viridis') 
legend_elements = list()
for i in range(len(Group2Label.keys())): 
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=Group2Label[i], 
        markerfacecolor=cmap(i / (len(Group2Label.keys())-1)), markersize=1))
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1,1), markerscale=10)
plt.tight_layout()
ax.grid(False)
plt.title('Lung 1')

#########################

datafile2 = '/Users/gabbut02/morpho-type/data/Lung3.csv'
data2 = pd.read_csv(datafile2, index_col='CellID')

data2['Group'] = [Label2Group[data2.loc[i, 'Label']] for i in range(1, np.shape(data2)[0]+1)]


features2 = data2[morphological_features]
features_norm2 = np.arcsinh((features2 )/ np.std(features2, 0))


umap_results2 = reducer.transform(features_norm2.values)

data2['x_umap'] = umap_results2[:,0]
data2['y_umap'] = umap_results2[:,1]

groups2 = data2.groupby('Label')

fig, ax = plt.subplots()
plt.scatter(data2.x_umap, data2.y_umap, c=data2.Group, s=0.1, cmap='viridis') 
legend_elements = list()
for i in range(len(Group2Label.keys())):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=Group2Label[i],
                          markerfacecolor=cmap(i / (len(Group2Label.keys())-1)), markersize=1))
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1,1), markerscale=10)
plt.tight_layout() 
ax.grid(False)
plt.title('Lung 3')

