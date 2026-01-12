# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:58:52 2024

@author: DELL
"""



import SimpleITK as sitk
import numpy as np

import os
import pandas as pd
from pandas import DataFrame as DF
import warnings
import time
from time import sleep
from tqdm import tqdm
from skimage import morphology,measure
import xlrd
from skimage.segmentation import slic
import json
import copy
from radiomics import featureextractor
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# import functools
# import concurrent.futures


 

def pixel_clustering(sub_mask, num, all_features, features, cluster):
    scaler = MinMaxScaler()
    all_features = scaler.fit_transform(all_features)
    features = scaler.transform(features)
    label_map = sub_mask.copy()
    kmeansCluster = KMeans(n_clusters=cluster).fit(all_features)
    clusters = kmeansCluster.predict(features)
    cnt = 0
    for k in range(1, num+1):
        label_map[sub_mask==k] = clusters[cnt] + 1
        cnt += 1
    return label_map
    
if __name__ == '__main__':
    start = time.perf_counter() 
    warnings.simplefilter('ignore')
    ## SHPH Dataset 
    data = xlrd.open_workbook('../../../Dataset/SHPH.xls')
    table = data.sheet_by_name(sheet_name="Sheet1")
    table_header = table.row_values(0)
    PatientIDs = table.col_values(0)[1:]
    PatientIDs = [str(num).strip().split('.')[0] for num in PatientIDs]
    
    n_mask_slic = 100
    cluster = 5
    SaveRoot = '../../../Dataset-1x1x1/SHPH'
    if not os.path.exists(SaveRoot):
        os.mkdir(SaveRoot)
    ImageRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/Image'
    MaskRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/Mask'
    ImageRoot_C1 = '../../../Dataset-1x1x1/SHPH/C1/Image'
    MaskRoot_C1 = '../../../Dataset-1x1x1/SHPH/C1/Mask'
    # Features Merge
    # all_features_BL = np.array([])
    # for ID in tqdm(PatientIDs):
    #     folder_bl = os.path.join(SaveRoot, 'BL')
    #     save_bl_SubFeature = os.path.join(folder_bl, 'SubFeature')
    #     BL_list = pd.read_csv(os.path.join(save_bl_SubFeature,ID+'.csv'))
    #     sub_features = BL_list.values[:,1:]
    #     feature_name = list(BL_list.head(0))[1:]
    #     if all_features_BL.size == 0:
    #         all_features_BL = sub_features
    #     else:
    #         all_features_BL = np.vstack((all_features_BL,sub_features))
    # all_BL = {}
    # for i in range(len(feature_name)):
    #     all_BL[feature_name[i]] = all_features_BL[:,i]
    # df = DF(all_BL).fillna('0')  
    # df.to_csv(os.path.join(folder_bl,'All_Feature_BL.csv'),index=False,sep=',') 
      
    # all_features_C1 = np.array([])
    # for ID in tqdm(PatientIDs):
    #     folder_C1 = os.path.join(SaveRoot, 'C1')
    #     save_C1_SubFeature = os.path.join(folder_C1, 'SubFeature')
    #     C1_list = pd.read_csv(os.path.join(save_C1_SubFeature,ID+'.csv'))
    #     sub_features = C1_list.values[:,1:]
    #     feature_name = list(C1_list.head(0))[1:]
    #     if all_features_C1.size == 0:
    #         all_features_C1 = sub_features
    #     else:
    #         all_features_C1 = np.vstack((all_features_C1,sub_features))
    # all_C1 = {}
    # for i in range(len(feature_name)):
    #     all_C1[feature_name[i]] = all_features_C1[:,i]
    # df = DF(all_C1).fillna('0')  
    # df.to_csv(os.path.join(folder_C1,'All_Feature_C1.csv'),index=False,sep=',')         
    
    # PatientIDs = ['N11942519']
    folder_bl = os.path.join(SaveRoot, 'BL')
    BL_list = pd.read_csv(os.path.join(folder_bl,'All_Feature_BL.csv'))
    all_feature_BL = BL_list.values
    folder_C1 = os.path.join(SaveRoot, 'C1')
    C1_list = pd.read_csv(os.path.join(folder_C1,'All_Feature_C1.csv'))
    all_feature_C1 = C1_list.values
    for ID in tqdm(PatientIDs):
        # ID = PatientIDs[i]
        
        save_bl_SlicMask = os.path.join(folder_bl, 'SlicMask')
        SlicMask = sitk.ReadImage(os.path.join(save_bl_SlicMask, ID+'.nii.gz')) 
        slic_mask = sitk.GetArrayFromImage(SlicMask)

        save_bl_SubFeature = os.path.join(folder_bl, 'SubFeature')
        BL_list = pd.read_csv(os.path.join(save_bl_SubFeature,ID+'.csv'))
        sub_features = BL_list.values[:,1:]
        num = sub_features.shape[0]
        
        sub_mask = pixel_clustering(slic_mask, num, all_feature_BL, sub_features, cluster)
        save_bl_SubMask = os.path.join(folder_bl, 'pcSubMask-'+str(cluster))
        if not os.path.exists(save_bl_SubMask):
            os.mkdir(save_bl_SubMask)
        SubMask = sitk.GetImageFromArray(sub_mask)
        SubMask.CopyInformation(SlicMask)
        sitk.WriteImage(SubMask, os.path.join(save_bl_SubMask, ID+'.nii.gz'))  
        

        save_C1_SlicMask = os.path.join(folder_C1, 'SlicMask')
        SlicMask = sitk.ReadImage(os.path.join(save_C1_SlicMask, ID+'.nii.gz')) 
        slic_mask = sitk.GetArrayFromImage(SlicMask)

        save_C1_SubFeature = os.path.join(folder_C1, 'SubFeature')
        C1_list = pd.read_csv(os.path.join(save_C1_SubFeature,ID+'.csv'))
        sub_features = C1_list.values[:,1:]
        num = sub_features.shape[0]
        sub_mask = pixel_clustering(slic_mask, num, all_feature_C1, sub_features, cluster)
        save_C1_SubMask = os.path.join(folder_C1, 'pcSubMask-'+str(cluster))
        if not os.path.exists(save_C1_SubMask):
            os.mkdir(save_C1_SubMask)
        SubMask = sitk.GetImageFromArray(sub_mask)
        SubMask.CopyInformation(SlicMask)
        sitk.WriteImage(SubMask, os.path.join(save_C1_SubMask, ID+'.nii.gz'))  
        
    end = time.perf_counter()
    print(end-start)