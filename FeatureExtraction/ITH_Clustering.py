# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 08:55:52 2024

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

def Extract_SubregionFeatures(image, mask, params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    # feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features
       
def slic_subregion(itk_image, mask, n_mask):
    image = sitk.GetArrayFromImage(itk_image)
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array = morphology.opening(mask_array,morphology.ball(1))
    subregions = slic(image, n_segments=n_mask,compactness=10.0,start_label=1, mask=mask_array, channel_axis=None)
    return subregions

def calculate_subregion_radiomics(image, mask, params_path='params_subregion.yaml'):
    mask_array = sitk.GetArrayFromImage(mask)
    num = mask_array.max()
    # _, num = measure.label(mask_array, connectivity=2, return_num=True) # meature label connected regions
    # partial_extract_feature_unit = functools.partial(Extract_SubregionFeatures, image)
    sub_featureVector = []
    for i in range(1, num + 1):
        section_mask = copy.deepcopy(mask_array)
        section_mask[section_mask != i] = 0 # split sv mask
        section_mask[section_mask == i] = 1 # split sv mask
        z,y,x = np.where(section_mask==1)
        if section_mask.sum()<5 or len(np.unique(y))+len(np.unique(z))+len(np.unique(x))<6:
            for zs in np.unique(z):
                section_mask[zs,np.unique(y).min()-1:np.unique(y).max()+1,np.unique(x).min()-1:np.unique(x).max()+1]=1
        section_mask_itk = sitk.GetImageFromArray(section_mask)
        section_mask_itk.CopyInformation(image)
        featureVector = {}
        featureVector['subNum'] = int(i)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=18) as executor:
        #     features = executor.map(partial_extract_feature_unit,section_mask_itk, params_path)
        features = Extract_SubregionFeatures(image,section_mask_itk, params_path)
        featureVector.update(features)
        sub_featureVector.append(featureVector)
    return sub_featureVector, num

def pixel_clustering(sub_mask, num, features, cluster):
    features = MinMaxScaler().fit_transform(features)
    label_map = sub_mask.copy()
    clusters = KMeans(n_clusters=cluster).fit_predict(features)
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
    cluster = 6
    SaveRoot = '../../../Dataset-1x1x1/SHPH'
    if not os.path.exists(SaveRoot):
        os.mkdir(SaveRoot)
    ImageRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/Image'
    MaskRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/Mask'
    ImageRoot_C1 = '../../../Dataset-1x1x1/SHPH/C1/Image'
    MaskRoot_C1 = '../../../Dataset-1x1x1/SHPH/C1/Mask'
    # PatientIDs = ['N11942519']
    for i in tqdm(range(len(PatientIDs))):
        ID = PatientIDs[i]
        Image = sitk.ReadImage(os.path.join(ImageRoot_BL, ID+'.nii.gz'))
        Mask = sitk.ReadImage(os.path.join(MaskRoot_BL, ID+'.nii.gz'))    
        slic_mask = slic_subregion(Image, Mask, n_mask=n_mask_slic).astype('uint16')
        folder_bl = os.path.join(SaveRoot, 'BL')
        if not os.path.exists(folder_bl):
            os.mkdir(folder_bl)
        save_bl_SlicMask = os.path.join(folder_bl, 'SlicMask')
        if not os.path.exists(save_bl_SlicMask):
            os.mkdir(save_bl_SlicMask)
        SlicMask = sitk.GetImageFromArray(slic_mask)
        SlicMask.CopyInformation(Image)
        sitk.WriteImage(SlicMask, os.path.join(save_bl_SlicMask, ID+'.nii.gz'))  
        
        sub_featureVector, num = calculate_subregion_radiomics(Image, SlicMask)
        df = DF(sub_featureVector).fillna('0')
        save_bl_SubFeature = os.path.join(folder_bl, 'SubFeature')
        if not os.path.exists(save_bl_SubFeature):
            os.mkdir(save_bl_SubFeature)
        df.to_csv(os.path.join(save_bl_SubFeature,ID+'.csv'),index=False,sep=',')
        sub_features = df.values[:,1:]
        sub_mask = pixel_clustering(slic_mask, num, sub_features, cluster)
        save_bl_SubMask = os.path.join(folder_bl, 'SubMask')
        if not os.path.exists(save_bl_SubMask):
            os.mkdir(save_bl_SubMask)
        SubMask = sitk.GetImageFromArray(sub_mask)
        SubMask.CopyInformation(Image)
        sitk.WriteImage(SubMask, os.path.join(save_bl_SubMask, ID+'.nii.gz'))  
        

        Image = sitk.ReadImage(os.path.join(ImageRoot_C1, ID+'.nii.gz'))
        Mask = sitk.ReadImage(os.path.join(MaskRoot_C1, ID+'.nii.gz'))    
        slic_mask = slic_subregion(Image, Mask, n_mask=n_mask_slic).astype('uint16')
        folder_C1 = os.path.join(SaveRoot, 'C1')
        if not os.path.exists(folder_C1):
            os.mkdir(folder_C1)
        save_C1_SlicMask = os.path.join(folder_C1, 'SlicMask')
        if not os.path.exists(save_C1_SlicMask):
            os.mkdir(save_C1_SlicMask)
        SlicMask = sitk.GetImageFromArray(slic_mask)
        SlicMask.CopyInformation(Image)
        sitk.WriteImage(SlicMask, os.path.join(save_C1_SlicMask, ID+'.nii.gz'))  
        
        sub_featureVector, num = calculate_subregion_radiomics(Image, SlicMask)
        df = DF(sub_featureVector).fillna('0')
        save_C1_SubFeature = os.path.join(folder_C1, 'SubFeature')
        if not os.path.exists(save_C1_SubFeature):
            os.mkdir(save_C1_SubFeature)
        df.to_csv(os.path.join(save_C1_SubFeature,ID+'.csv'),index=False,sep=',')
        sub_features = df.values[:,1:]
        sub_mask = pixel_clustering(slic_mask, num, sub_features, cluster)
        save_C1_SubMask = os.path.join(folder_C1, 'SubMask')
        if not os.path.exists(save_C1_SubMask):
            os.mkdir(save_C1_SubMask)
        SubMask = sitk.GetImageFromArray(sub_mask)
        SubMask.CopyInformation(Image)
        sitk.WriteImage(SubMask, os.path.join(save_C1_SubMask, ID+'.nii.gz'))  
        
    end = time.perf_counter()
    print(end-start)