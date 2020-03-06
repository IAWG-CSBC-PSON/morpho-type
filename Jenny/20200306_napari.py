# 2020-03-05 Vanderbuilt Hackathon: Cell Morphology Challenge
# author engje
# ipython --gui=qt
#%gui qt
import napari
import os
import skimage
import numpy as np
import copy
import pandas as pd


#paths
codedir = 'C:\\Users\\engje\\Desktop\\HACKATHON'
regdir = f'{codedir}\Images'

ls_cell = ['Immune', 'Stroma', 'Tumor']

ls_sample = ['LUNG2'] #'LUNG3','LUNG1','LUNG2',

#dictionaries
d_img = {'LUNG1':'LUNG-1-LN_40X.ome.tif',
 'LUNG2':'LUNG-2-BR_40X.ome.tif',
 'LUNG3':'LUNG-3-PR_40X.ome.tif'}

d_seg = {'LUNG1':'LUNG-1-LN_40X_Seg_labeled.tif',
 'LUNG2':'LUNG-2-BR_40X_Seg_labeled.tif',
 'LUNG3':'LUNG-3-PR_40X_Seg_labeled.tif'}

#baseline
d_base = {'LUNG1':'predictions/Lung1-xgboost.csv',
 'LUNG2':'predictions/Lung2-xgboost.csv',
 'LUNG3':'predictions/Lung3-xgboost.csv',
 }

d_data = {'LUNG1':'Lung1.csv',
 'LUNG2':'Lung2.csv',
 'LUNG3':'Lung3.csv',
 }
#Yues tunes parameters 
d_base = {'LUNG1':'morpho-type/predictions/python_xgboost/xgboost_Lung1_morpho.pred.csv',
 'LUNG2':'morpho-type/predictions/python_xgboost/xgboost_Lung2_morpho.pred.csv',
 'LUNG3':'morpho-type/predictions/python_xgboost/xgboost_Lung3_morpho.pred.csv',
 }

#Yues tunes parameters on DAPI 
d_dapi = {'LUNG1':'morpho-type/predictions/python_xgboost/xgboost_Lung1_morpho_dapi.pred.csv',
 'LUNG2':'morpho-type/predictions/python_xgboost/xgboost_Lung2_morpho_dapi.pred.csv',
 'LUNG3':'morpho-type/predictions/python_xgboost/xgboost_Lung3_morpho_dapi.pred.csv',
 }

#Yues neighborhood
d_neigh = {'LUNG1':'morpho-type/predictions/python_xgboost/xgboost_Lung1_morpho_dapi_neighbor.pred.csv',
 'LUNG2':'morpho-type/predictions/python_xgboost/xgboost_Lung2_morpho_dapi_neighbor.pred.csv',
 'LUNG3':'morpho-type/predictions/python_xgboost/xgboost_Lung3_morpho_dapi_neighbor.pred.csv',
 }


for s_sample in ls_sample:
    #load ground truth 
    df_pos = pd.read_csv(f'{codedir}/data/{d_data[s_sample]}',index_col=0)
    #load img
    img = skimage.io.imread(f'{regdir}/{d_img[s_sample]}')
    # load segmentation
    label_image = skimage.io.imread(f'{regdir}/{d_seg[s_sample]}')
    #load baseline xgboost
    df_predict = pd.read_csv(f'{codedir}/{d_base[s_sample]}', index_col=0)
    df_pos = df_pos.loc[df_predict.index]
    #load DAPI xgboost
    df_dapi = pd.read_csv(f'{codedir}/{d_dapi[s_sample]}', index_col=0)
    #load neighbor xgboost
    df_neigh = pd.read_csv(f'{codedir}/{d_neigh[s_sample]}', index_col=0)
    #create a Viewer and add round 2 dapi image
    viewer = napari.view_image(img[4], name='DAPI',rgb=False,blending='additive',colormap='blue')
    #LUNG1 has problem with some segmentation labels, filter out
    if s_sample == 'LUNG1':
        label_image[label_image > 87500] = 0
        '''
        df_mis = pd.read_csv(f'{codedir}/misclassified.csv')
        label_image_cell = copy.deepcopy(label_image)
        label_image_cell[~np.isin(label_image_cell, df_pos.index)] = 0
        label_image_cell[~np.isin(label_image_cell, df_mis.CellID)] = 0
        df_out = pd.read_csv(f'{codedir}/outliers.csv')
        label_image_cell = copy.deepcopy(label_image)
        label_image_cell[~np.isin(label_image_cell, df_pos.index)] = 0
        label_image_cell[~np.isin(label_image_cell, df_out.CellID)] = 0
        viewer.add_labels(label_image_cell, name=f'UMAP_out')
        '''
    #add all cells segmentation labels
    viewer.add_labels(label_image, name='all_seg')
    #add Keratin, asma, CD45
    viewer.add_image(img[14],name='KERATIN',rgb=False,blending='additive',colormap='cyan')
    viewer.add_image(img[22],name='CD45',rgb=False,blending='additive',colormap='yellow')
    viewer.add_image(img[34],name='ASMA',rgb=False,blending='additive',colormap='magenta')
    #label each cell segmentation basin

    #ground truth
    s_marker= 'Immune'
    #for s_marker in ls_cell:
    label_image_cell = copy.deepcopy(label_image)
    label_image_cell[~np.isin(label_image_cell, df_pos.index)] = 0
    label_image_cell[~np.isin(label_image_cell, df_pos[df_pos.Label==s_marker].index)] = 0
    viewer.add_labels(label_image_cell, name=f'{s_marker}_seg')
    
    # baseline predictions
    label_image_cell = copy.deepcopy(label_image)
    label_image_cell[~np.isin(label_image_cell, df_predict.index)] = 0
    #opposite of where the labels are predicted positive
    label_image_cell[~np.isin(label_image_cell, df_predict[df_predict.Pred==s_marker].index)] = 0
    viewer.add_labels(label_image_cell, name=f'{s_marker}_baseline')
    
    #dapi predictions
    label_image_cell = copy.deepcopy(label_image)
    label_image_cell[~np.isin(label_image_cell, df_dapi.index)] = 0
    #opposite of where the labels are predicted positive
    label_image_cell[~np.isin(label_image_cell, df_dapi[df_dapi.Pred==s_marker].index)] = 0
    viewer.add_labels(label_image_cell, name=f'{s_marker}_dapi')
    
    #neighbor predictions
    label_image_cell = copy.deepcopy(label_image)
    label_image_cell[~np.isin(label_image_cell, df_neigh.index)] = 0
    #opposite of where the labels are predicted positive
    label_image_cell[~np.isin(label_image_cell, df_neigh[df_neigh.Pred==s_marker].index)] = 0
    viewer.add_labels(label_image_cell, name=f'{s_marker}_neigh')
    break
