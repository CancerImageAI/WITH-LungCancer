# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:09:55 2024

@author: DELL
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc,confusion_matrix
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import SelectFdr, f_classif, chi2, SelectFromModel,mutual_info_classif,SelectPercentile
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, LogisticRegression,LinearRegression
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
import scipy.stats as stats
from pandas import DataFrame as DF
from imblearn.over_sampling import SMOTE
import xlrd
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test 
from lifelines import CoxPHFitter
import os
import SimpleITK as sitk
from comparision_auc_delong import delong_roc_test
import seaborn as sns
import warnings

def train_val_split(Class):
    train_rate = 0.8
    # val_rate = 0.3
    negative_list = [i for i in range(len(Class)) if Class[i]==0]
    np.random.seed(42)
    np.random.shuffle(negative_list)
    positive_list = [i for i in range(len(Class)) if Class[i]==1]
    np.random.shuffle(positive_list)

    negative_train_len = int(len(negative_list) * train_rate)
    positive_train_len = int(len(positive_list) * train_rate)
    negative_train_list = negative_list[:negative_train_len]
    positive_train_list = positive_list[:positive_train_len]
    negative_val_list = negative_list[negative_train_len:]
    positive_val_list = positive_list[positive_train_len:]
    train_list = negative_train_list + positive_train_list
    val_list = negative_val_list + positive_val_list
    return train_list, val_list

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

def prediction_score(truth, predicted):
    TN, FP, FN, TP = confusion_matrix(truth, predicted, labels=[0,1]).ravel()
    print(TN, FP, FN, TP)
    ACC = (TP+TN)/(TN+FP+FN+TP)
    SEN = TP/(FN+TP)
    SPE = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    print('ACC:',ACC)
    print('Sensitivity:',SEN)
    print('Specifity:',SPE)
    print('PPV:',PPV)
    print('NPV:',NPV)
    OR = (TP*TN)/(FP*FN)
    print('OR:',OR)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
   y = TPR - FPR
   Youden_index = np.argmax(y)  # Only the first occurrence is returned.
   optimal_threshold = threshold[Youden_index]
   point = [FPR[Youden_index], TPR[Youden_index]]
   return optimal_threshold, point

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    font = {'family' : 'Times New Roman',
 			'weight' : 'normal',
 			'size'   : 12,}
    plt.rc('font', **font)
    ## BL  
    BL_path = '../../../Result/WholeTumorFeature_BL.csv'
    BL_list = pd.read_csv(BL_path)
    tag = np.any(BL_list.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        BL_list = BL_list.fillna(BL_list.median())
    pass
    BL_PatientID = list(np.array(BL_list['PatientID']))
    BL_Feature = np.array(BL_list.values[:,:-2])
    BL_FeatureName = np.array(list(BL_list.head(0))[:-2])
    Delta_FeatureName = ['Intra-'+i for i in BL_FeatureName]
    BL_FeatureName = ['Intra-'+i for i in BL_FeatureName]

    Class = np.array(BL_list['PR_Status']).astype(int)
    
    ## C1  
    C1_path = '../../../Result/WholeTumorFeature_C1.csv'
    C1_list = pd.read_csv(C1_path)
    tag = np.any(C1_list.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        C1_list = C1_list.fillna(C1_list.median())
    pass
    C1_PatientID = list(np.array(C1_list['PatientID']))
    C1_Feature = np.array(C1_list.values[:,:-2])
    C1_FeatureName = np.array(list(C1_list.head(0))[:-2])
    C1_FeatureName = ['Intra-'+i for i in C1_FeatureName]
    ind_C1 = [C1_PatientID.index(i) for i in BL_PatientID]
    C1_Feature = C1_Feature[ind_C1, :]
    
    ind_train, ind_test = train_val_split(Class)
    BL_Feature_train = BL_Feature[ind_train, :]
    C1_Feature_train = C1_Feature[ind_train, :]
    Class_train = Class[ind_train]
    PatientID_train = np.array(BL_PatientID)[ind_train]
    Train_Result = {}
    Train_Result['ID'] = PatientID_train
    Train_Result['Class'] = Class_train
    
    BL_Feature_test = BL_Feature[ind_test, :]
    C1_Feature_test = C1_Feature[ind_test, :]
    Class_test = Class[ind_test]
    PatientID_test = np.array(BL_PatientID)[ind_test]
    Test_Result = {}
    Test_Result['ID'] = PatientID_test
    Test_Result['Class'] = Class_test

    ## FUSCC
    ## BL  
    BL_path_FUSCC = '../../../Result/WholeTumorFeature_BL_FUSCC.csv'
    BL_list_FUSCC = pd.read_csv(BL_path_FUSCC)
    tag = np.any(BL_list_FUSCC.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        BL_list_FUSCC = BL_list_FUSCC.fillna(BL_list_FUSCC.median())
    pass
    BL_PatientID_FUSCC = list(np.array(BL_list_FUSCC['PatientID']))
    BL_Feature_FUSCC = np.array(BL_list_FUSCC.values[:,:-2])


    Class_FUSCC = np.array(BL_list_FUSCC['PR_Status']).astype(int)
    
    ## C1  
    C1_path_FUSCC = '../../../Result/WholeTumorFeature_C1_FUSCC.csv'
    C1_list_FUSCC = pd.read_csv(C1_path_FUSCC)
    tag = np.any(C1_list_FUSCC.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        C1_list_FUSCC = C1_list_FUSCC.fillna(C1_list_FUSCC.median())
    pass
    C1_PatientID_FUSCC = list(np.array(C1_list_FUSCC['PatientID']))
    C1_Feature_FUSCC = np.array(C1_list_FUSCC.values[:,:-2])
    ind_C1_FUSCC = [C1_PatientID_FUSCC.index(i) for i in BL_PatientID_FUSCC]
    C1_Feature_FUSCC = C1_Feature_FUSCC[ind_C1_FUSCC, :]
    
    PatientID_FUSCC = np.array(BL_PatientID_FUSCC)
    FUSCC_Result = {}
    FUSCC_Result['ID'] = PatientID_FUSCC
    FUSCC_Result['Class'] = Class_FUSCC
    
    ## FUZSH
    ## BL  
    BL_path_FUZSH = '../../../Result/WholeTumorFeature_BL_FUZSH.csv'
    BL_list_FUZSH = pd.read_csv(BL_path_FUZSH)
    tag = np.any(BL_list_FUZSH.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        BL_list_FUZSH = BL_list_FUZSH.fillna(BL_list_FUZSH.median())
    pass
    BL_PatientID_FUZSH = list(np.array(BL_list_FUZSH['PatientID']))
    BL_Feature_FUZSH = np.array(BL_list_FUZSH.values[:,:-2])


    Class_FUZSH = np.array(BL_list_FUZSH['PR_Status']).astype(int)
    
    ## C1  
    C1_path_FUZSH = '../../../Result/WholeTumorFeature_C1_FUZSH.csv'
    C1_list_FUZSH = pd.read_csv(C1_path_FUZSH)
    tag = np.any(C1_list_FUZSH.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        C1_list_FUZSH = C1_list_FUZSH.fillna(C1_list_FUZSH.median())
    pass
    C1_PatientID_FUZSH = list(np.array(C1_list_FUZSH['PatientID']))
    C1_Feature_FUZSH = np.array(C1_list_FUZSH.values[:,:-2])
    ind_C1_FUZSH = [C1_PatientID_FUZSH.index(i) for i in BL_PatientID_FUZSH]
    C1_Feature_FUZSH = C1_Feature_FUZSH[ind_C1_FUZSH, :]
    
    PatientID_FUZSH = np.array(BL_PatientID_FUZSH)
    FUZSH_Result = {}
    FUZSH_Result['ID'] = PatientID_FUZSH
    FUZSH_Result['Class'] = Class_FUZSH
    
    # BL Model
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    _ = scaler.fit_transform(np.vstack((np.vstack((BL_Feature_train,BL_Feature_test)),np.vstack((BL_Feature_FUSCC,BL_Feature_FUZSH)))))

    x_train_BL = scaler.transform(np.array(BL_Feature_train))
    x_test_BL = scaler.transform(BL_Feature_test)
    x_FUSCC_BL = scaler.transform(BL_Feature_FUSCC)
    x_FUZSH_BL = scaler.transform(BL_Feature_FUZSH)

    pre_selector = SelectPercentile(score_func=f_classif, percentile=3)
    # _ = pre_selector.fit_transform(np.vstack((np.vstack((x_train_BL,x_test_BL)),np.vstack((x_FUSCC_BL,x_FUZSH_BL)))),
    #                                Class_train)

    x_train_BL = pre_selector.fit_transform(np.array(x_train_BL),Class_train)
    x_test_BL = pre_selector.transform(x_test_BL)
    x_FUSCC_BL = pre_selector.transform(x_FUSCC_BL)
    x_FUZSH_BL = pre_selector.transform(x_FUZSH_BL)

    # FDR_selector = SelectFdr(score_func=f_classif,alpha=0.1).fit(np.array(x_train_BL), Class_train)
    # x_train_BL = FDR_selector.transform(np.array(x_train_BL))
    # x_test_BL = FDR_selector.transform(x_test_BL)
    # x_FUSCC_BL = FDR_selector.transform(x_FUSCC_BL)
    
    BL_Selected_Name_in = pre_selector.get_feature_names_out(BL_FeatureName)
    
    estimator_BL = linear_model.Lasso(alpha=0.01,random_state=0)

    selector_Img = RFE(estimator_BL, n_features_to_select=10, step=1)#5
    # selector_Img = KernelPCA(n_components=i,random_state=0)
    train_BL = selector_Img.fit_transform(x_train_BL,Class_train)
    test_BL = selector_Img.transform(x_test_BL)
    FUSCC_BL = selector_Img.transform(x_FUSCC_BL)
    FUZSH_BL = selector_Img.transform(x_FUZSH_BL)

    BL_Selected_Name = selector_Img.get_feature_names_out(BL_Selected_Name_in)
    print(BL_Selected_Name)

    
    indices = list(np.where(selector_Img.support_==True)[0])
    SelectedFeatures = np.array(BL_FeatureName)[indices]
    
    selected_feature = x_train_BL[:,indices]
    feature_names = ['F'+str(i) for i in range(1,11)]
    Class_Type = [i if i==1  else 'Non-MPR' for i in Class_train]
    Class_Type = [i if i=='Non-MPR' else 'MPR' for i in Class_Type]
    selectedFeature = {}
    selectedFeature['FeatureName'] = np.ravel([[i]*len(selected_feature) for i in feature_names])
    selectedFeature['Feature'] = np.ravel([selected_feature[:,i] for i in range(selected_feature.shape[1])])
    selectedFeature['Class_Type'] = np.ravel([Class_Type for i in range(selected_feature.shape[1])])
    selectedFeature = pd.DataFrame.from_dict(selectedFeature)
    
    font = {'family' : 'Arial',
            'weight' :  'medium',
            'size'   : 10,}
    plt.rc('font', **font)
    plt.figure(figsize=(6,4))
    sns.boxenplot(x="FeatureName", y="Feature",hue='Class_Type',data=selectedFeature,
                      palette='Paired_r')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='',edgecolor='k',loc='upper right')
    plt.subplots_adjust(top=0.995,bottom=0.06,left=0.05,right=0.995,hspace=0,wspace=0)
    
    x_BL, y_BL = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train_BL, Class_train)
    clf_BL = XGBClassifier(max_depth=3,learning_rate=1e-5,n_estimators=10,objective='binary:logistic',random_state=0)

    # clf_BL = svm.SVC(kernel="rbf", probability=True, random_state=0)
    # clf_BL = BaggingClassifier(base_estimator=svm.SVC(kernel="rbf", probability=True, random_state=0),
    #                 n_estimators=10, random_state=0)
    clf_BL.fit(x_BL, y_BL)
    train_prob_BL = clf_BL.predict_proba(train_BL)[:,1]
    pred_label_train_BL = clf_BL.predict(train_BL)
    pred_label_train_BL = np.array(pred_label_train_BL).astype(int)
    fpr_train_BL,tpr_train_BL,threshold_train_BL = roc_curve(Class_train, np.array(train_prob_BL)) ###计算真正率和假正率
    auc_score_train_BL = auc(fpr_train_BL,tpr_train_BL)
    auc_l_train_BL, auc_h_train_BL, auc_std_train_BL = confindence_interval_compute(np.array(train_prob_BL), Class_train)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_BL,auc_std_train_BL),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_BL, auc_h_train_BL))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(Class_train,pred_label_train_BL)*100)) 
    # prediction_score(Class_train, pred_label_train_BL)
    print('-----------------------------------------------')
    Train_Result['BL_Score'] = train_prob_BL

    test_prob_BL = clf_BL.predict_proba(test_BL)[:,1]
    pred_label_BL = clf_BL.predict(test_BL)
    pred_label_BL = np.array(pred_label_BL).astype(int)
    fpr_BL,tpr_BL,threshold_BL = roc_curve(Class_test, np.array(test_prob_BL)) ###计算真正率和假正率
    auc_score_BL = auc(fpr_BL,tpr_BL)
    auc_l_BL, auc_h_BL, auc_std_BL = confindence_interval_compute(np.array(test_prob_BL), Class_test)
    print('Testing Dataset AUC:%.2f+/-%.2f'%(auc_score_BL,auc_std_BL),'  95%% CI:[%.2f,%.2f]'%(auc_l_BL,auc_h_BL))
    print('Testing Dataset  ACC:%.2f%%'%(accuracy_score(Class_test,pred_label_BL)*100)) 
    # TN, FP, FN, TP = confusion_matrix(test_Class, pred_label_BL, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_test, pred_label_BL)
    print('-----------------------------------------------')
    Test_Result['BL_Score'] = test_prob_BL

    FUSCC_prob_BL = clf_BL.predict_proba(FUSCC_BL)[:,1]
    pred_label_BL = clf_BL.predict(FUSCC_BL)
    pred_label_BL = np.array(pred_label_BL).astype(int)
    fpr_BL,tpr_BL,threshold_BL = roc_curve(Class_FUSCC, np.array(FUSCC_prob_BL)) ###计算真正率和假正率
    auc_score_BL = auc(fpr_BL,tpr_BL)
    auc_l_BL, auc_h_BL, auc_std_BL = confindence_interval_compute(np.array(FUSCC_prob_BL), Class_FUSCC)
    print('FUSCC Dataset AUC:%.2f+/-%.2f'%(auc_score_BL,auc_std_BL),'  95%% CI:[%.2f,%.2f]'%(auc_l_BL,auc_h_BL))
    print('FUSCC Dataset  ACC:%.2f%%'%(accuracy_score(Class_FUSCC,pred_label_BL)*100)) 
    # TN, FP, FN, TP = confusion_matrix(FUSCC_Class, pred_label_BL, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_FUSCC, pred_label_BL)
    print('-----------------------------------------------')
    FUSCC_Result['BL_Score'] = FUSCC_prob_BL

    FUZSH_prob_BL = clf_BL.predict_proba(FUZSH_BL)[:,0]
    pred_label_BL = 1-clf_BL.predict(FUZSH_BL)
    pred_label_BL = np.array(pred_label_BL).astype(int)
    fpr_BL,tpr_BL,threshold_BL = roc_curve(Class_FUZSH, np.array(FUZSH_prob_BL)) ###计算真正率和假正率
    auc_score_BL = auc(fpr_BL,tpr_BL)
    auc_l_BL, auc_h_BL, auc_std_BL = confindence_interval_compute(np.array(FUZSH_prob_BL), Class_FUZSH)
    print('FUZSH Dataset AUC:%.2f+/-%.2f'%(auc_score_BL,auc_std_BL),'  95%% CI:[%.2f,%.2f]'%(auc_l_BL,auc_h_BL))
    print('FUZSH Dataset  ACC:%.2f%%'%(accuracy_score(Class_FUZSH,pred_label_BL)*100)) 
    # TN, FP, FN, TP = confusion_matrix(FUZSH_Class, pred_label_BL, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_FUZSH, pred_label_BL)
    print('-----------------------------------------------')
    FUZSH_Result['BL_Score'] = FUZSH_prob_BL
    

    # C1 Model
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    _ = scaler.fit_transform(np.vstack((np.vstack((C1_Feature_train,C1_Feature_test)),np.vstack((C1_Feature_FUSCC,C1_Feature_FUZSH)))))

    x_train_C1 = scaler.transform(np.array(C1_Feature_train))
    x_test_C1 = scaler.transform(C1_Feature_test)
    x_FUSCC_C1 = scaler.transform(C1_Feature_FUSCC)
    x_FUZSH_C1 = scaler.transform(C1_Feature_FUZSH)

    pre_selector = SelectPercentile(score_func=f_classif, percentile=8)
    x_train_C1 = pre_selector.fit_transform(np.array(x_train_C1),Class_train)
    x_test_C1 = pre_selector.transform(x_test_C1)
    x_FUSCC_C1 = pre_selector.transform(x_FUSCC_C1)
    x_FUZSH_C1 = pre_selector.transform(x_FUZSH_C1)

    C1_Selected_Name = pre_selector.get_feature_names_out(C1_FeatureName)

    estimator_C1 = linear_model.Lasso(alpha=0.01,random_state=0)

    # estimator_C1 = Ridge(alpha=0.01,random_state=0)
    # estimator_C1 = SVC(kernel="rbf",random_state=0)
    selector_Img = RFE(estimator_C1, n_features_to_select=10, step=1)#14
    # selector_Img = KernelPCA(n_components=i,random_state=0)
    train_C1 = selector_Img.fit_transform(x_train_C1,Class_train)
    test_C1 = selector_Img.transform(x_test_C1)
    FUSCC_C1 = selector_Img.transform(x_FUSCC_C1)
    FUZSH_C1 = selector_Img.transform(x_FUZSH_C1)

    C1_Selected_Name = selector_Img.get_feature_names_out(C1_Selected_Name)
    print(C1_Selected_Name)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    SelectedFeatures = np.array(C1_FeatureName)[indices]
    
    selected_feature = x_train_C1[:,indices]
    feature_names = ['F'+str(i) for i in range(1,11)]
    Class_Type = [i if i==1  else 'Non-MPR' for i in Class_train]
    Class_Type = [i if i=='Non-MPR' else 'MPR' for i in Class_Type]
    selectedFeature = {}
    selectedFeature['FeatureName'] = np.ravel([[i]*len(selected_feature) for i in feature_names])
    selectedFeature['Feature'] = np.ravel([selected_feature[:,i] for i in range(selected_feature.shape[1])])
    selectedFeature['Class_Type'] = np.ravel([Class_Type for i in range(selected_feature.shape[1])])
    selectedFeature = pd.DataFrame.from_dict(selectedFeature)
    
    font = {'family' : 'Arial',
            'weight' :  'medium',
            'size'   : 10,}
    plt.rc('font', **font)
    plt.figure(figsize=(6,4))
    sns.boxenplot(x="FeatureName", y="Feature",hue='Class_Type',data=selectedFeature,
                      palette='Paired_r')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='',edgecolor='k',loc='upper right')
    plt.subplots_adjust(top=0.995,bottom=0.06,left=0.05,right=0.995,hspace=0,wspace=0)
    
    x_C1, y_C1 = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train_C1, Class_train)

    # clf_C1 = svm.SVC(kernel="rbf", probability=True, random_state=0)
    clf_C1 = XGBClassifier(max_depth=3,learning_rate=1e-5,n_estimators=10,objective='binary:logistic',random_state=0)

    # clf_C1 = BaggingClassifier(base_estimator=svm.SVC(kernel="rbf", probability=True, random_state=0),
    #                 n_estimators=10, random_state=0)
    clf_C1.fit(x_C1, y_C1)
    train_prob_C1 = clf_C1.predict_proba(train_C1)[:,1]
    pred_label_train_C1 = clf_C1.predict(train_C1)
    pred_label_train_C1 = np.array(pred_label_train_C1).astype(int)
    fpr_train_C1,tpr_train_C1,threshold_train_C1 = roc_curve(Class_train, np.array(train_prob_C1)) ###计算真正率和假正率
    auc_score_train_C1 = auc(fpr_train_C1,tpr_train_C1)
    auc_l_train_C1, auc_h_train_C1, auc_std_train_C1 = confindence_interval_compute(np.array(train_prob_C1), Class_train)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_C1,auc_std_train_C1),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_C1, auc_h_train_C1))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(Class_train,pred_label_train_C1)*100)) 
    # prediction_score(Class_train, pred_label_train_C1)
    print('-----------------------------------------------')
    Train_Result['C1_Score'] = train_prob_C1

    test_prob_C1 = clf_C1.predict_proba(test_C1)[:,1]
    pred_label_C1 = clf_C1.predict(test_C1)
    pred_label_C1 = np.array(pred_label_C1).astype(int)
    fpr_C1,tpr_C1,threshold_C1 = roc_curve(Class_test, np.array(test_prob_C1)) ###计算真正率和假正率
    auc_score_C1 = auc(fpr_C1,tpr_C1)
    auc_l_C1, auc_h_C1, auc_std_C1 = confindence_interval_compute(np.array(test_prob_C1), Class_test)
    print('Testing Dataset AUC:%.2f+/-%.2f'%(auc_score_C1,auc_std_C1),'  95%% CI:[%.2f,%.2f]'%(auc_l_C1,auc_h_C1))
    print('Testing Dataset  ACC:%.2f%%'%(accuracy_score(Class_test,pred_label_C1)*100)) 
    # TN, FP, FN, TP = confusion_matrix(test_Class, pred_label_C1, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_test, pred_label_C1)
    print('-----------------------------------------------')
    Test_Result['C1_Score'] = test_prob_C1

    FUSCC_prob_C1 = clf_C1.predict_proba(FUSCC_C1)[:,1]
    pred_label_C1 = clf_C1.predict(FUSCC_C1)
    pred_label_C1 = np.array(pred_label_C1).astype(int)
    fpr_C1,tpr_C1,threshold_C1 = roc_curve(Class_FUSCC, np.array(FUSCC_prob_C1)) ###计算真正率和假正率
    auc_score_C1 = auc(fpr_C1,tpr_C1)
    auc_l_C1, auc_h_C1, auc_std_C1 = confindence_interval_compute(np.array(FUSCC_prob_C1), Class_FUSCC)
    print('FUSCC Dataset AUC:%.2f+/-%.2f'%(auc_score_C1,auc_std_C1),'  95%% CI:[%.2f,%.2f]'%(auc_l_C1,auc_h_C1))
    print('FUSCC Dataset  ACC:%.2f%%'%(accuracy_score(Class_FUSCC,pred_label_C1)*100)) 
    # TN, FP, FN, TP = confusion_matrix(FUSCC_Class, pred_label_C1, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_FUSCC, pred_label_C1)
    print('-----------------------------------------------')
    FUSCC_Result['C1_Score'] = FUSCC_prob_C1

    FUZSH_prob_C1 = clf_C1.predict_proba(FUZSH_C1)[:,0]
    pred_label_C1 = 1-clf_C1.predict(FUZSH_C1)
    pred_label_C1 = np.array(pred_label_C1).astype(int)
    fpr_C1,tpr_C1,threshold_C1 = roc_curve(Class_FUZSH, np.array(FUZSH_prob_C1)) ###计算真正率和假正率
    auc_score_C1 = auc(fpr_C1,tpr_C1)
    auc_l_C1, auc_h_C1, auc_std_C1 = confindence_interval_compute(np.array(FUZSH_prob_C1), Class_FUZSH)
    print('FUZSH Dataset AUC:%.2f+/-%.2f'%(auc_score_C1,auc_std_C1),'  95%% CI:[%.2f,%.2f]'%(auc_l_C1,auc_h_C1))
    print('FUZSH Dataset  ACC:%.2f%%'%(accuracy_score(Class_FUZSH,pred_label_C1)*100)) 
    # TN, FP, FN, TP = confusion_matrix(FUZSH_Class, pred_label_C1, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_FUZSH, pred_label_C1)
    print('-----------------------------------------------')
    FUZSH_Result['C1_Score'] = FUZSH_prob_C1
    
    # Delta Model
    Delta_Feature_train = (C1_Feature_train-BL_Feature_train)/(BL_Feature_train+1e-100)
    Delta_Feature_test = (C1_Feature_test-BL_Feature_test)/(BL_Feature_test+1e-100)
    Delta_Feature_FUSCC = (C1_Feature_FUSCC-BL_Feature_FUSCC)/(BL_Feature_FUSCC+1e-100)
    Delta_Feature_FUZSH = (C1_Feature_FUZSH-BL_Feature_FUZSH)/(BL_Feature_FUZSH+1e-100)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    _ = scaler.fit_transform(np.vstack((np.vstack((np.vstack((Delta_Feature_train,Delta_Feature_test)),Delta_Feature_FUSCC)),Delta_Feature_FUZSH)))
    x_train_Delta = scaler.transform(np.array(Delta_Feature_train))
    x_test_Delta = scaler.transform(Delta_Feature_test)
    x_FUSCC_Delta = scaler.transform(Delta_Feature_FUSCC)
    x_FUZSH_Delta = scaler.transform(Delta_Feature_FUZSH)

    pre_selector = SelectPercentile(score_func=f_classif, percentile=12)
    x_train_Delta = pre_selector.fit_transform(np.array(x_train_Delta),Class_train)
    x_test_Delta = pre_selector.transform(x_test_Delta)
    x_FUSCC_Delta = pre_selector.transform(x_FUSCC_Delta)
    x_FUZSH_Delta = pre_selector.transform(x_FUZSH_Delta)

    Delta_Selected_Name_in = pre_selector.get_feature_names_out(Delta_FeatureName)
    
    estimator_Delta = linear_model.Lasso(alpha=0.01,random_state=0)#5
    # estimator_Delta = Ridge(alpha=0.01,random_state=0)
    # estimator_Delta = SVC(kernel="rbf",random_state=0)
    selector_Img = RFE(estimator_Delta, n_features_to_select=8, step=60)#15
    # selector_Img = KernelPCA(n_components=i,random_state=0)
    train_Delta = selector_Img.fit_transform(x_train_Delta,Class_train)
    test_Delta = selector_Img.transform(x_test_Delta)
    FUSCC_Delta = selector_Img.transform(x_FUSCC_Delta)
    FUZSH_Delta = selector_Img.transform(x_FUZSH_Delta)

    Delta_Selected_Name = selector_Img.get_feature_names_out(Delta_Selected_Name_in)
    print(Delta_Selected_Name)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    SelectedFeatures = np.array(Delta_Selected_Name_in)[indices]
    print(SelectedFeatures)
    selected_feature = x_train_Delta[:,indices]
    feature_names = ['F'+str(i) for i in range(1,9)]
    Class_Type = [i if i==1  else 'Non-MPR' for i in Class_train]
    Class_Type = [i if i=='Non-MPR' else 'MPR' for i in Class_Type]
    selectedFeature = {}
    selectedFeature['FeatureName'] = np.ravel([[i]*len(selected_feature) for i in feature_names])
    selectedFeature['Feature'] = np.ravel([selected_feature[:,i] for i in range(selected_feature.shape[1])])
    selectedFeature['Class_Type'] = np.ravel([Class_Type for i in range(selected_feature.shape[1])])
    selectedFeature = pd.DataFrame.from_dict(selectedFeature)
    
    font = {'family' : 'Arial',
            'weight' :  'medium',
            'size'   : 10,}
    plt.rc('font', **font)
    plt.figure(figsize=(6,4))
    sns.boxenplot(x="FeatureName", y="Feature",hue='Class_Type',data=selectedFeature,
                      palette='Paired_r')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='',edgecolor='k',loc='upper right')
    plt.subplots_adjust(top=0.995,bottom=0.06,left=0.05,right=0.995,hspace=0,wspace=0)
    
    x_Delta, y_Delta = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train_Delta, Class_train)

    clf_Delta = XGBClassifier(max_depth=3,learning_rate=1e-5,n_estimators=10,objective='binary:logistic',random_state=0)

    # clf_Delta = BaggingClassifier(base_estimator=svm.SVC(kernel="rbf", probability=True, random_state=0),
    #                 n_estimators=10, random_state=0)
    clf_Delta.fit(x_Delta, y_Delta)
    train_prob_Delta = clf_Delta.predict_proba(train_Delta)[:,1]
    pred_label_train_Delta = clf_Delta.predict(train_Delta)
    pred_label_train_Delta = np.array(pred_label_train_Delta).astype(int)
    fpr_train_Delta,tpr_train_Delta,threshold_train_Delta = roc_curve(Class_train, np.array(train_prob_Delta)) ###计算真正率和假正率
    auc_score_train_Delta = auc(fpr_train_Delta,tpr_train_Delta)
    auc_l_train_Delta, auc_h_train_Delta, auc_std_train_Delta = confindence_interval_compute(np.array(train_prob_Delta), Class_train)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_Delta,auc_std_train_Delta),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_Delta, auc_h_train_Delta))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(Class_train,pred_label_train_Delta)*100)) 
    # prediction_score(Class_train, pred_label_train_Delta)
    print('-----------------------------------------------')
    Train_Result['Delta_Score'] = train_prob_Delta

    
    test_prob_Delta = clf_Delta.predict_proba(test_Delta)[:,1]
    pred_label_Delta = clf_Delta.predict(test_Delta)
    pred_label_Delta = np.array(pred_label_Delta).astype(int)
    fpr_Delta,tpr_Delta,threshold_Delta = roc_curve(Class_test, np.array(test_prob_Delta)) ###计算真正率和假正率
    auc_score_Delta = auc(fpr_Delta,tpr_Delta)
    auc_l_Delta, auc_h_Delta, auc_std_Delta = confindence_interval_compute(np.array(test_prob_Delta), Class_test)
    print('Testing Dataset AUC:%.2f+/-%.2f'%(auc_score_Delta,auc_std_Delta),'  95%% CI:[%.2f,%.2f]'%(auc_l_Delta,auc_h_Delta))
    print('Testing Dataset  ACC:%.2f%%'%(accuracy_score(Class_test,pred_label_Delta)*100)) 
    # TN, FP, FN, TP = confusion_matrix(test_Class, pred_label_Delta, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_test, pred_label_Delta)
    print('-----------------------------------------------')
    Test_Result['Delta_Score'] = test_prob_Delta


    FUSCC_prob_Delta = clf_Delta.predict_proba(FUSCC_Delta)[:,1]
    pred_label_Delta = clf_Delta.predict(FUSCC_Delta)
    pred_label_Delta = np.array(pred_label_Delta).astype(int)
    fpr_Delta,tpr_Delta,threshold_Delta = roc_curve(Class_FUSCC, np.array(FUSCC_prob_Delta)) ###计算真正率和假正率
    auc_score_Delta = auc(fpr_Delta,tpr_Delta)
    auc_l_Delta, auc_h_Delta, auc_std_Delta = confindence_interval_compute(np.array(FUSCC_prob_Delta), Class_FUSCC)
    print('FUSCC Dataset AUC:%.2f+/-%.2f'%(auc_score_Delta,auc_std_Delta),'  95%% CI:[%.2f,%.2f]'%(auc_l_Delta,auc_h_Delta))
    print('FUSCC Dataset  ACC:%.2f%%'%(accuracy_score(Class_FUSCC,pred_label_Delta)*100)) 
    # TN, FP, FN, TP = confusion_matrix(FUSCC_Class, pred_label_Delta, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_FUSCC, pred_label_Delta)
    print('-----------------------------------------------')
    FUSCC_Result['Delta_Score'] = FUSCC_prob_Delta


    FUZSH_prob_Delta = clf_Delta.predict_proba(FUZSH_Delta)[:,1]
    pred_label_Delta = clf_Delta.predict(FUZSH_Delta)
    pred_label_Delta = np.array(pred_label_Delta).astype(int)
    fpr_Delta,tpr_Delta,threshold_Delta = roc_curve(Class_FUZSH, np.array(FUZSH_prob_Delta)) ###计算真正率和假正率
    auc_score_Delta = auc(fpr_Delta,tpr_Delta)
    auc_l_Delta, auc_h_Delta, auc_std_Delta = confindence_interval_compute(np.array(FUZSH_prob_Delta), Class_FUZSH)
    print('FUZSH Dataset AUC:%.2f+/-%.2f'%(auc_score_Delta,auc_std_Delta),'  95%% CI:[%.2f,%.2f]'%(auc_l_Delta,auc_h_Delta))
    print('FUZSH Dataset  ACC:%.2f%%'%(accuracy_score(Class_FUZSH,pred_label_Delta)*100)) 
    # TN, FP, FN, TP = confusion_matrix(FUZSH_Class, pred_label_Delta, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    # prediction_score(Class_FUZSH, pred_label_Delta)
    print('-----------------------------------------------')
    FUZSH_Result['Delta_Score'] = FUZSH_prob_Delta


    # Fusion Model
    Fusion_Feature_train = np.vstack((train_prob_BL,train_prob_C1,train_prob_Delta))
    Fusion_Feature_test = np.vstack((test_prob_BL,test_prob_C1,test_prob_Delta))
    Fusion_Feature_FUSCC = np.vstack((FUSCC_prob_BL,FUSCC_prob_C1,FUSCC_prob_Delta))
    Fusion_Feature_FUZSH = np.vstack((FUZSH_prob_BL,FUZSH_prob_C1,FUZSH_prob_Delta))

    train_Fusion = Fusion_Feature_train.transpose((1,0))
    test_Fusion = Fusion_Feature_test.transpose((1,0))
    FUSCC_Fusion = Fusion_Feature_FUSCC.transpose((1,0))
    FUZSH_Fusion = Fusion_Feature_FUZSH.transpose((1,0))
    x_Fusion, y_Fusion = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train_Fusion, Class_train)
    # clf_Fusion = svm.SVC(kernel="linear", probability=True, random_state=0)
    clf_Fusion = XGBClassifier(max_depth=5,learning_rate=1e-6,n_estimators=10,objective='binary:logistic',random_state=0)

    # clf_Fusion = BaggingClassifier(base_estimator=svm.SVC(kernel="linear", probability=True, random_state=0),
    #                 n_estimators=10, random_state=0)
    clf_Fusion.fit(x_Fusion, y_Fusion)
    train_prob_Fusion = clf_Fusion.predict_proba(train_Fusion)[:,1]
    pred_label_train_Fusion = clf_Fusion.predict(train_Fusion)
    pred_label_train_Fusion = np.array(pred_label_train_Fusion).astype(int)
    fpr_train_Fusion,tpr_train_Fusion,threshold_train_Fusion = roc_curve(Class_train, np.array(train_prob_Fusion)) ###计算真正率和假正率
    auc_score_train_Fusion = auc(fpr_train_Fusion,tpr_train_Fusion)
    auc_l_train_Fusion, auc_h_train_Fusion, auc_std_train_Fusion = confindence_interval_compute(np.array(train_prob_Fusion), Class_train)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_Fusion,auc_std_train_Fusion),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_Fusion, auc_h_train_Fusion))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(Class_train,pred_label_train_Fusion)*100)) 
    # prediction_score(Class_train, pred_label_train_Fusion)
    print('-----------------------------------------------')

    # Train_Result['Fusion_Score'] = train_prob_Fusion
    # train_df = pd.DataFrame(Train_Result)
    # train_df.to_csv(r'../result/Wholetumor_Result_Training.csv')
    
    test_prob_Fusion = clf_Fusion.predict_proba(test_Fusion)[:,1]
    pred_label_test_Fusion = clf_Fusion.predict(test_Fusion)
    pred_label_test_Fusion = np.array(pred_label_test_Fusion).astype(int)
    fpr_test_Fusion,tpr_test_Fusion,threshold_test_Fusion = roc_curve(Class_test, np.array(test_prob_Fusion)) ###计算真正率和假正率
    auc_score_test_Fusion = auc(fpr_test_Fusion,tpr_test_Fusion)
    auc_l_test_Fusion, auc_h_test_Fusion, auc_std_test_Fusion = confindence_interval_compute(np.array(test_prob_Fusion), Class_test)
    print('Testing Dataset AUC:%.2f+/-%.2f'%(auc_score_test_Fusion,auc_std_test_Fusion),'  95%% CI:[%.2f,%.2f]'%(auc_l_test_Fusion, auc_h_test_Fusion))
    print('Testing Dataset ACC:%.2f%%'%(accuracy_score(Class_test,pred_label_test_Fusion)*100)) 
    # prediction_score(Class_test, pred_label_test_Fusion)
    print('-----------------------------------------------')

    # Test_Result['Fusion_Score'] = test_prob_Fusion
    # test_df = pd.DataFrame(Test_Result)
    # test_df.to_csv(r'../result/Wholetumor_Result_Testing.csv')
    
    FUSCC_prob_Fusion = clf_Fusion.predict_proba(FUSCC_Fusion)[:,1]
    
    pred_label_FUSCC_Fusion = clf_Fusion.predict(FUSCC_Fusion)
    pred_label_FUSCC_Fusion = np.array(pred_label_FUSCC_Fusion).astype(int)
    fpr_FUSCC_Fusion,tpr_FUSCC_Fusion,threshold_FUSCC_Fusion = roc_curve(Class_FUSCC, np.array(FUSCC_prob_Fusion)) ###计算真正率和假正率
    auc_score_FUSCC_Fusion = auc(fpr_FUSCC_Fusion,tpr_FUSCC_Fusion)
    auc_l_FUSCC_Fusion, auc_h_FUSCC_Fusion, auc_std_FUSCC_Fusion = confindence_interval_compute(np.array(FUSCC_prob_Fusion), Class_FUSCC)
    print('Validation Dataset AUC:%.2f+/-%.2f'%(auc_score_FUSCC_Fusion,auc_std_FUSCC_Fusion),'  95%% CI:[%.2f,%.2f]'%(auc_l_FUSCC_Fusion, auc_h_FUSCC_Fusion))
    print('Validation Dataset ACC:%.2f%%'%(accuracy_score(Class_FUSCC,pred_label_FUSCC_Fusion)*100)) 
    # prediction_score(Class_FUSCC, pred_label_FUSCC_Fusion)
    print('-----------------------------------------------')
    
    # FUSCC_Result['Fusion_Score'] = FUSCC_prob_Fusion
    # FUSCC_df = pd.DataFrame(FUSCC_Result)
    # FUSCC_df.to_csv(r'../result/Wholetumor_Result_FUSCC.csv')
    
    FUZSH_prob_Fusion = clf_Fusion.predict_proba(FUZSH_Fusion)[:,1]
    pred_label_FUZSH_Fusion = clf_Fusion.predict(FUZSH_Fusion)
    pred_label_FUZSH_Fusion = np.array(pred_label_FUZSH_Fusion).astype(int)
    fpr_FUZSH_Fusion,tpr_FUZSH_Fusion,threshold_FUZSH_Fusion = roc_curve(Class_FUZSH, np.array(FUZSH_prob_Fusion)) ###计算真正率和假正率
    auc_score_FUZSH_Fusion = auc(fpr_FUZSH_Fusion,tpr_FUZSH_Fusion)
    auc_l_FUZSH_Fusion, auc_h_FUZSH_Fusion, auc_std_FUZSH_Fusion = confindence_interval_compute(np.array(FUZSH_prob_Fusion), Class_FUZSH)
    print('Validation Dataset AUC:%.2f+/-%.2f'%(auc_score_FUZSH_Fusion,auc_std_FUZSH_Fusion),'  95%% CI:[%.2f,%.2f]'%(auc_l_FUZSH_Fusion, auc_h_FUZSH_Fusion))
    print('Validation Dataset ACC:%.2f%%'%(accuracy_score(Class_FUZSH,pred_label_FUZSH_Fusion)*100)) 
    # prediction_score(Class_FUZSH, pred_label_FUZSH_Fusion)
    print('-----------------------------------------------')
    # FUZSH_Result['Fusion_Score'] = FUZSH_prob_Fusion
    # FUZSH_df = pd.DataFrame(FUZSH_Result)
    # FUZSH_df.to_csv(r'../result/Wholetumor_Result_FUZSH.csv')


   