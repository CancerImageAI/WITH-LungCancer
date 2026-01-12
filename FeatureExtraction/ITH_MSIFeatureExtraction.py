# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:37:48 2024

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
from radiomics.firstorder import RadiomicsFirstOrder
import math

def generate_MSI_array(mask_array):
    region_num = mask_array.max()
    MSI_array = np.zeros((region_num+1, region_num+1))
    z,x,y = np.where(mask_array>0)
    
    for i in range(len(z)):
        ind_c = mask_array[z[i],x[i],y[i]]
        pos_x = [x[i]-1, x[i], x[i]+1]
        pos_y = [y[i]-1, y[i], y[i]+1]
        for pt_x in pos_x:
            for pt_y in pos_y:
                if pt_x != x[i] and pt_y != y[i]:
                    # print(z[i],pt_x,pt_y)
                    ind_n = mask_array[z[i],pt_x,pt_y]
                    MSI_array[ind_c, ind_n] = MSI_array[ind_c, ind_n]+1
            
    MSI_array[0,0] = 0
    for i in range(MSI_array.shape[0]):
        MSI_array[0,i] = MSI_array[i,0]
    return MSI_array, region_num+1
    
    
def cal_ITH_feature(MSI_array,gray_level):
    Cont, Entr, Asm, Hom, Ener, Diss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Cont += (i-j)*(i-j)*MSI_array[i][j]
            Asm += MSI_array[i][j]*MSI_array[i][j]
            Ener += math.sqrt(MSI_array[i][j]*MSI_array[i][j])
            Diss += MSI_array[i][j]*abs(i-j)
            Hom += MSI_array[i][j]/(1+(i-j)*(i-j))
            if MSI_array[i][j]>0.0:
                Entr += MSI_array[i][j]*math.log(MSI_array[i][j])
    firstorderExtractor = RadiomicsFirstOrder(sitk.GetImageFromArray(MSI_array),sitk.GetImageFromArray((MSI_array>0).astype('uint8')))
    features = firstorderExtractor.execute()
    features = {'firstorder_'+k: v for k,v in features.items()}   
    features['MSI_contrast'] = Cont
    features['MSI_dissimilarity'] = Diss
    features['MSI_homogeneity'] = Hom
    features['MSI_ASM'] = Asm
    features['MSI_energy'] = Ener
    features['MSI_entroy'] = -Entr
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
    MaskRoot_BL = '../../../Dataset-1x1x1/SHPH/BL/pcSubMask-5'   
    MaskRoot_C1 = '../../../Dataset-1x1x1/SHPH/C1/pcSubMask-5'
    SHPH_Feature_BL = []
    SHPH_Feature_C1 = []
    for i in tqdm(range(len(PatientIDs))):
        try:
            ID = PatientIDs[i]
            Mask_BL = sitk.ReadImage(os.path.join(MaskRoot_BL, ID+'.nii.gz'))
            Mask_BL_array = sitk.GetArrayFromImage(Mask_BL)
            MSI_array_BL,level_BL = generate_MSI_array(Mask_BL_array)
            feature_bl = cal_ITH_feature(MSI_array_BL,level_BL)
            feature_bl['PatientID'] = ID
            feature_bl['PR_Status']  = PR_Status[i]    
            SHPH_Feature_BL.append(feature_bl)
    
            Mask_C1 = sitk.ReadImage(os.path.join(MaskRoot_C1, ID+'.nii.gz'))
            Mask_C1_array = sitk.GetArrayFromImage(Mask_C1)
            MSI_array_C1,level_C1 = generate_MSI_array(Mask_C1_array)
            feature_C1 = cal_ITH_feature(MSI_array_C1,level_C1)
            feature_C1['PatientID'] = ID
            feature_C1['PR_Status']  = PR_Status[i]    
            SHPH_Feature_C1.append(feature_C1)             

        except:
            print(ID)
    df = DF(SHPH_Feature_BL).fillna('0')
    df.to_csv('../../../Result/ITH_pcMSIFeature_BL-5.csv', index = False, sep=',')
    df = DF(SHPH_Feature_C1).fillna('0')
    df.to_csv('../../../Result/ITH_pcMSIFeature_C1-5.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)