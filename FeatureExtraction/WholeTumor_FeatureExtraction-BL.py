# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:42:05 2023

@author: DELL
"""



import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import os
from pandas import DataFrame as DF
import time
from tqdm import tqdm
import random
import xlrd


def Extract_Features(image,mask):
    paramsFile = os.path.abspath('params.yaml')
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
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    start = time.perf_counter()
    ## SHPH Dataset 
    data = xlrd.open_workbook('../../../Dataset/SHPH.xls')
    table = data.sheet_by_name(sheet_name="Sheet1")
    table_header = table.row_values(0)
    PatientIDs = table.col_values(0)[1:]
    PatientIDs = [str(num).strip().split('.')[0] for num in PatientIDs]
    PR_Status = table.col_values(19)[1:]
    PR_Status = [1 if int(i)<=2 else 0 for i in PR_Status]

    ImageRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/Image'
    MaskRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/Mask'
    SHPH_Feature = []
    for i in tqdm(range(len(PatientIDs))):
        ID = PatientIDs[i]
        Image = sitk.ReadImage(os.path.join(ImageRoot_BL, ID+'.nii.gz'))
        Mask = sitk.ReadImage(os.path.join(MaskRoot_BL, ID+'.nii.gz'))
                               
        feature, feature_info = Extract_Features(Image, Mask) 
        feature['PatientID'] = ID
        feature['PR_Status']  = PR_Status[i]    
        SHPH_Feature.append(feature)
    df = DF(SHPH_Feature).fillna('0')
    df.to_csv('../../../Result/WholeTumorFeature_BL.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)
    
