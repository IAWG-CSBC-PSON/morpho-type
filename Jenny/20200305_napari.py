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

ls_sample = ['LUNG2','LUNG3'] #'LUNG1',

#dictionaries
d_img = {'LUNG1':'LUNG-1-LN_40X.ome.tif',
 'LUNG2':'LUNG-2-BR_40X.ome.tif',
 'LUNG3':'LUNG3-PR_40X.ome.tif'}

d_seg = {'LUNG1':'LUNG-1-LN_40X_Seg_labeled.tif',
 'LUNG2':'LUNG-2-BR_40X_Seg_labeled.tif',
 'LUNG3':'LUNG-3-PR_40X_Seg_labeled.tif'}
 
d_base = {'LUNG1':'Lung1-xgboost.csv',
 'LUNG2':'Lung2-xgboost.csv',
 'LUNG3':'Lung3-xgboost.csv',
 }

d_data = {'LUNG1':'Lung1.csv',
 'LUNG2':'Lung2.csv',
 'LUNG3':'Lung3.csv',
 }

for s_sample in ls_sample:
    #load ground truth 
    df_pos = pd.read_csv(f'{codedir}/data/{d_data[s_sample]}',index_col=0)
    #load img
    img = skimage.io.imread(f'{regdir}/{d_img[s_sample]}')
    # load segmentation
    label_image = skimage.io.imread(f'{regdir}/{d_seg[s_sample]}')
    #load baseline xgboost
    df_predict = pd.read_csv(f'{codedir}/predictions/{d_base[s_sample]}', index_col=0)
    df_predict['Predict_ID'] = f'baseline_{s_sample}'
    #create a Viewer and add round 2 dapi image
    viewer = napari.view_image(img[4], name='DAPI',rgb=False,blending='additive',colormap='blue')
    #LUNG1 has problem with some segmentation labels, filter out
    if s_sample == 'LUNG1':
        label_image[label_image > 87500] = 0
        df_mis = pd.read_csv(f'{codedir}/misclassified.csv')
    #add all cells segmentation labels
    viewer.add_labels(label_image, name='all_seg')
    #add Keratin, asma, CD45
    viewer.add_image(img[14],name='KERATIN',rgb=False,blending='additive',colormap='cyan')
    viewer.add_image(img[22],name='CD45',rgb=False,blending='additive',colormap='yellow')
    viewer.add_image(img[34],name='ASMA',rgb=False,blending='additive',colormap='magenta')
    #label each cell segmentation basin

    #ground truth
    for s_marker in ls_cell:
        label_image_cell = copy.deepcopy(label_image)
        label_image_cell[~np.isin(label_image_cell, df_pos.index)] = 0
        label_image_cell[~np.isin(label_image_cell, df_pos[df_pos.Label==s_marker].index)] = 0
        viewer.add_labels(label_image_cell, name=f'{s_marker}_seg')

    #baseline predictions
    for s_marker in ls_cell:
        label_image_cell = copy.deepcopy(label_image)
        label_image_cell[~np.isin(label_image_cell, df_predict.index)] = 0
        label_image_cell[~np.isin(label_image_cell, df_predict[df_predict.Pred==s_marker].index)] = 0
        viewer.add_labels(label_image_cell, name=f'{s_marker}_baseline')

    #wrong baseline predictions only
    #predicted positive for a marker but negative (false positive)
    for s_marker in ls_cell:
        label_image_cell = copy.deepcopy(label_image)
        label_image_cell[(~np.isin(label_image_cell, df_predict[df_predict.Pred==s_marker].index)) & (~np.isin(label_image_cell, df_pos[df_pos.Label==s_marker].index))] = 0

    break

#OLD

#### LUNG-1 image ####
'''
### read data ###
df_pos1 = pd.read_csv(f'{codedir}/data/Lung1.csv',index_col=0)

#read img
img_dapi1 = skimage.io.imread(f'{regdir}/{ls_img[0]}')

#cell seg
label_image1 = skimage.io.imread(f'{regdir}/{ls_seg[0]}')

# create a Viewer and add an image here
viewer = napari.view_image(img_dapi1[4], name='DAPI',rgb=False,blending='additive')

#get rid of extra labels
label_image1[~np.isin(label_image1, df_pos1.index)] = 0
label_image1[label_image1 > 87500] = 0
viewer.add_labels(label_image1, name='all_seg')

#marker images
viewer.add_image(img_dapi1[14],name='KERATIN',rgb=False,blending='additive')
viewer.add_image(img_dapi1[22],name='CD45',rgb=False,blending='additive')
viewer.add_image(img_dapi1[34],name='ASMA',rgb=False,blending='additive')

#label each cell segmentation basin
#tumor
for s_marker in ls_cell:
    label_image_cell = copy.deepcopy(label_image1)
    label_image_cell[~np.isin(label_image_cell, df_pos1.index)] = 0
    label_image_cell[~np.isin(label_image_cell, df_pos1[df_pos1.Label==s_marker].index)] = 0
    viewer.add_labels(label_image_cell, name=f'{s_marker}_seg')

'''

#### LUNG-2 image ####
'''
#read data
df_pos2 = pd.read_csv(f'{codedir}/data/Lung2.csv',index_col=0)

#read img
img_dapi2 = skimage.io.imread(f'{regdir}/{ls_img[1]}')

#cell seg
label_image2 = skimage.io.imread(f'{regdir}/{ls_seg[1]}')

# create a Viewer and add an image here
viewer = napari.view_image(img_dapi2[4], name='DAPI',rgb=False,blending='additive')
viewer.add_labels(label_image2, name='all_seg')

#marker images
viewer.add_image(img_dapi2[14],name='KERATIN',rgb=False,blending='additive')
viewer.add_image(img_dapi2[22],name='CD45',rgb=False,blending='additive')
viewer.add_image(img_dapi2[34],name='ASMA',rgb=False,blending='additive')

#label each cell segmentation basin
#truth
for s_marker in ls_cell:
    label_image_cell = copy.deepcopy(label_image2)
    label_image_cell[~np.isin(label_image_cell, df_pos2.index)] = 0
    label_image_cell[~np.isin(label_image_cell, df_pos2[df_pos2.Label==s_marker].index)] = 0
    viewer.add_labels(label_image_cell, name=f'{s_marker}_seg')

#baseline predictions
for s_marker in ls_cell:
    label_image_cell = copy.deepcopy(label_image2)
    label_image_cell[~np.isin(label_image_cell, df_predict.index)] = 0
    label_image_cell[~np.isin(label_image_cell, df_predict[df_predict.Pred==s_marker].index)] = 0
    viewer.add_labels(label_image_cell, name=f'{s_marker}_baseline')

#wrong baseline predictions only
#predicted positive for a marker but negative (false positive)
for s_marker in ls_cell:
    label_image_cell = copy.deepcopy(label_image2)
    label_image_cell[~np.isin(label_image_cell, df_pos2[df_pos2.Label==s_marker].index)] = 0
    label_image_cell[~np.isin(label_image_cell, df_predict[df_predict.Pred==s_marker].index)] = 0

#predicted negative but positive (false negative)
'''


'''
############################################
#### LUNG-3 image ####
img_dapi = skimage.io.imread(f'{regdir}/{ls_img[2]}')

#cell seg
label_image = skimage.io.imread(f'{regdir}/{ls_seg[2]}')

# create a Viewer and add an image here
viewer = napari.view_image(img_dapi[43], name='hello',rgb=False,blending='additive')
viewer.add_labels(label_image, name='all_seg')

#label each cell type
#tumor
li_index = df_pos[df_pos.Label=='Tumor'].index
label_image_cell = copy.deepcopy(label_image)
label_image_cell[~np.isin(label_image_cell, li_index)] = 0
viewer.add_labels(label_image_cell, name='tumor_seg')
'''
