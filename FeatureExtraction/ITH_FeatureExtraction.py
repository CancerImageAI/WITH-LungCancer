# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:57:32 2024

@author: DELL
"""
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from pandas import DataFrame as DF
from tqdm import tqdm
import time
from skimage import morphology,measure
import xlrd

def calITHscore(label_map, min_area=200, thresh=1):
    """
    Calculate ITHscore from clustering label map
    Args:
        label_map: Numpy array. Clustering label map
        min_area: Int. For tumor area (pixels) smaller than "min_area", we don't consider connected-region smaller than "thresh"
        thresh: Int. The threshold for connected-region's area, only valid when tumor area < min_area
    Returns:
        ith_score: Float. The level of ITH, between 0 and 1
    """
    size = np.sum(label_map > 0)  # Record the number of total pixels
    num_regions_list = []
    max_area_list = []
    for i in np.unique(label_map)[1:]:  # For each gray level except 0 (background)
        flag = 1  # Flag to count this gray level, in case this gray level has only one pixel
        # Find (8-) connected-components. "num_regions" is the number of connected components
        labeled, num_regions = measure.label(label_map==i, connectivity=2, return_num=True)
        max_area = 0
        for j in np.unique(labeled)[1:]:  # 0 is background (here is all the other regions)
            # Ignore the region with only 1 or "thresh" px
            if size <= min_area:
                if np.sum(labeled == j) <= thresh:
                    num_regions -= 1
                    if num_regions == 0:  # In case there is only one region
                        flag = 0
                else:
                    temp_area = np.sum(labeled == j)
                    if temp_area > max_area:
                        max_area = temp_area
            else:
                if np.sum(labeled == j) <= 1:
                    num_regions -= 1
                    if num_regions == 0:  # In case there is only one region
                        flag = 0
                else:
                    temp_area = np.sum(labeled == j)
                    if temp_area > max_area:
                        max_area = temp_area
        if flag == 1:
            num_regions_list.append(num_regions)
            max_area_list.append(max_area)
    # Calculate the ITH score
    ith_score = 0
    # num_size = 0
    # print(num_regions_list)
    for k in range(len(num_regions_list)):
        ith_score += float(max_area_list[k]) / num_regions_list[k]
        # num_size += num_regions_list[k]
    # Normalize each area with total size
    ith_score = ith_score / size#*num_size)
    ith_score = 1 - ith_score

    return ith_score

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
            Mask_BL_array = sitk.GetArrayFromImage(Mask_BL)
            ith_score_BL = calITHscore(Mask_BL_array, min_area=250, thresh=1)
            feature_bl = {}
            feature_bl['PatientID'] = ID
            feature_bl['PR_Status']  = PR_Status[i]    
            feature_bl['ITH_Score'] = ith_score_BL
            SHPH_Feature_BL.append(feature_bl)

            Mask_C1 = sitk.ReadImage(os.path.join(MaskRoot_C1, ID+'.nii.gz'))
            Mask_C1_array = sitk.GetArrayFromImage(Mask_C1)
            ith_score_C1 = calITHscore(Mask_C1_array, min_area=250, thresh=1)
            feature_C1 = {}
            feature_C1['PatientID'] = ID
            feature_C1['PR_Status']  = PR_Status[i]    
            feature_C1['ITH_Score'] = ith_score_C1
            SHPH_Feature_C1.append(feature_C1)             

        except:
            print(ID)
    df = DF(SHPH_Feature_BL).fillna('0')
    df.to_csv('../../../Result/ITH_Score_BL-5.csv', index = False, sep=',')
    df = DF(SHPH_Feature_C1).fillna('0')
    df.to_csv('../../../Result/ITH_Score_C1-5.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)