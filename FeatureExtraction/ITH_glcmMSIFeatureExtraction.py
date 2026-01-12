# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:49:04 2024

@author: DELL
"""

import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from pandas import DataFrame as DF
from tqdm import tqdm
import time
from skimage import feature
import xlrd
from radiomics.glcm import RadiomicsGLCM
import math

def cal_glcmITH_feature(mask_image):
    mask_array = sitk.GetArrayFromImage(mask_image)
    glcmExtractor = RadiomicsGLCM(mask_image, sitk.GetImageFromArray((mask_array>0).astype('uint8')))
    features = glcmExtractor.execute()
    features = {'MSI_'+k: v for k,v in features.items()}  
    return features


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
    MaskRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/SubMask-5'   
    MaskRoot_C1 = '../../../Dataset-1x1x1/SHPH/C1/SubMask-5'
    SHPH_Feature_BL = []
    SHPH_Feature_C1 = []
    for i in tqdm(range(len(PatientIDs))):
        try:
            ID = PatientIDs[i]
            Mask_BL = sitk.ReadImage(os.path.join(MaskRoot_BL, ID+'.nii.gz'))
            feature_bl = cal_glcmITH_feature(Mask_BL)
            feature_bl['PatientID'] = ID
            feature_bl['PR_Status']  = PR_Status[i]    
            SHPH_Feature_BL.append(feature_bl)
    
            Mask_C1 = sitk.ReadImage(os.path.join(MaskRoot_C1, ID+'.nii.gz'))
            feature_C1 = cal_glcmITH_feature(Mask_C1)
            feature_C1['PatientID'] = ID
            feature_C1['PR_Status']  = PR_Status[i]    
            SHPH_Feature_C1.append(feature_C1)             

        except:
            print(ID)
    df = DF(SHPH_Feature_BL).fillna('0')
    df.to_csv('../../../Result/ITH_glcmMSIFeature_BL-5.csv', index = False, sep=',')
    df = DF(SHPH_Feature_C1).fillna('0')
    df.to_csv('../../../Result/ITH_glcmMSIFeature_C1-5.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)