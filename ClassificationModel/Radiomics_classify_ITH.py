# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:54:10 2024

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
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel,mutual_info_classif,SelectPercentile
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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test 
from lifelines import CoxPHFitter
import os
import SimpleITK as sitk
from comparision_auc_delong import delong_roc_test
import seaborn as sns

def train_val_split(Class):
    train_rate = 0.7
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
    font = {'family' : 'Times New Roman',
 			'weight' : 'normal',
 			'size'   : 12,}
    plt.rc('font', **font)
    ## BL  
    BL_path = '../../../Result/ITH_Score_BL.csv'
    BL_list = pd.read_csv(BL_path)
    tag = np.any(BL_list.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        BL_list = BL_list.fillna(BL_list.median())
    pass
    BL_PatientID = list(np.array(BL_list['PatientID']))
    BL_Feature = np.array(BL_list.values[:,-1])
    Class = np.array(BL_list['PR_Status']).astype(int)
    
    ## C1  
    C1_path = '../../../Result/ITH_Score_C1.csv'
    C1_list = pd.read_csv(C1_path)
    tag = np.any(C1_list.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        C1_list = C1_list.fillna(C1_list.median())
    pass
    C1_PatientID = list(np.array(C1_list['PatientID']))
    C1_Feature = np.array(C1_list.values[:,-1])
    ind_C1 = [C1_PatientID.index(i) for i in BL_PatientID]
    C1_Feature = C1_Feature[ind_C1]
    
    ind_train, ind_test = train_val_split(Class)
    BL_Feature_train = BL_Feature[ind_train]
    C1_Feature_train = C1_Feature[ind_train]
    Class_train = Class[ind_train]
    PatientID_train = np.array(BL_PatientID)[ind_train]
    Train_Result = {}
    Train_Result['ID'] = PatientID_train
    Train_Result['Class'] = Class_train
    
    BL_Feature_test = BL_Feature[ind_test]
    C1_Feature_test = C1_Feature[ind_test]
    Class_test = Class[ind_test]
    PatientID_test = np.array(BL_PatientID)[ind_test]
    Test_Result = {}
    Test_Result['ID'] = PatientID_test
    Test_Result['Class'] = Class_test

    # BL Model
    # thresh_BL=0.5
    train_prob_BL = BL_Feature_train
    fpr_train_BL,tpr_train_BL,threshold_train_BL = roc_curve(Class_train, np.array(train_prob_BL)) ###计算真正率和假正率
    thresh_BL,_ = Find_Optimal_Cutoff(tpr_train_BL,fpr_train_BL, threshold_train_BL)
    pred_label_train_BL = np.array(train_prob_BL>thresh_BL).astype(int)
    auc_score_train_BL = auc(fpr_train_BL,tpr_train_BL)
    auc_l_train_BL, auc_h_train_BL, auc_std_train_BL = confindence_interval_compute(np.array(train_prob_BL), Class_train)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_BL,auc_std_train_BL),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_BL, auc_h_train_BL))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(Class_train,pred_label_train_BL)*100)) 
    # prediction_score(Class_train, pred_label_train_BL)
    print('-----------------------------------------------')
    Train_Result['BL_Score'] = train_prob_BL

    test_prob_BL = BL_Feature_test
    pred_label_BL = np.array(test_prob_BL>thresh_BL).astype(int)
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

    # thresh_C1 = 0.5
    # C1 Model
    train_prob_C1 = C1_Feature_train
    fpr_train_C1,tpr_train_C1,threshold_train_C1 = roc_curve(Class_train, np.array(train_prob_C1)) ###计算真正率和假正率
    thresh_C1,_ = Find_Optimal_Cutoff(tpr_train_C1,fpr_train_C1, threshold_train_C1)
    pred_label_train_C1 = np.array(train_prob_C1>thresh_C1).astype(int)
    auc_score_train_C1 = auc(fpr_train_C1,tpr_train_C1)
    auc_l_train_C1, auc_h_train_C1, auc_std_train_C1 = confindence_interval_compute(np.array(train_prob_C1), Class_train)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_C1,auc_std_train_C1),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_C1, auc_h_train_C1))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(Class_train,pred_label_train_C1)*100)) 
    # prediction_score(Class_train, pred_label_train_C1)
    print('-----------------------------------------------')
    Train_Result['C1_Score'] = train_prob_C1

    test_prob_C1 = C1_Feature_test
    pred_label_C1 = np.array(test_prob_C1>thresh_C1).astype(int)
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
    
    
    # Delta Model
    Delta_Feature_train = (C1_Feature_train-BL_Feature_train)
    Delta_Feature_test = (C1_Feature_test-BL_Feature_test)
    # thresh_Delta = 0.5
    train_prob_Delta = Delta_Feature_train
    fpr_train_Delta,tpr_train_Delta,threshold_train_Delta = roc_curve(Class_train, np.array(train_prob_Delta)) ###计算真正率和假正率
    thresh_Delta,_ = Find_Optimal_Cutoff(tpr_train_Delta,fpr_train_Delta, threshold_train_Delta)
    pred_label_train_Delta = np.array(train_prob_Delta>thresh_Delta).astype(int)
    auc_score_train_Delta = auc(fpr_train_Delta,tpr_train_Delta)
    auc_l_train_Delta, auc_h_train_Delta, auc_std_train_Delta = confindence_interval_compute(np.array(train_prob_Delta), Class_train)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_Delta,auc_std_train_Delta),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_Delta, auc_h_train_Delta))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(Class_train,pred_label_train_Delta)*100)) 
    # prediction_score(Class_train, pred_label_train_Delta)
    print('-----------------------------------------------')
    Train_Result['Delta_Score'] = train_prob_Delta
    train_df = pd.DataFrame(Train_Result)
    train_df.to_csv(r'../result/ITH_Result_Training.csv')
    
    test_prob_Delta = Delta_Feature_test
    pred_label_Delta = np.array(test_prob_Delta>thresh_Delta).astype(int)
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
    test_df = pd.DataFrame(Test_Result)
    test_df.to_csv(r'../result/ITH_Result_Testing.csv')
    


    lw = 1.5
    plt.figure(figsize=(5,5)) 
    plt.plot(fpr_train_Delta,tpr_train_Delta, color='r',linestyle='-',
              lw=lw, label='Delta ITH\nAUC=%.2f'%auc_score_train_Delta+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_train_Delta, auc_l_train_Delta, auc_h_train_Delta))
    plt.plot(fpr_train_BL,tpr_train_BL, color='b',linestyle='-',
              lw=lw, label='BL ITH\nAUC=%.2f'%auc_score_train_BL+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_train_BL, auc_l_train_BL, auc_h_train_BL))
    plt.plot(fpr_train_C1,tpr_train_C1, color='g',linestyle='-',
              lw=lw, label='C1 ITH\nAUC=%.2f'%auc_score_train_C1+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_train_C1, auc_l_train_C1, auc_h_train_C1))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    plt.legend(loc="lower right",edgecolor='k',title='Training Cohort',fontsize=10,fancybox=False)
    plt.subplots_adjust(top=0.985,bottom=0.095,left=0.115,right=0.975,hspace=0,wspace=0)
    
    plt.figure(figsize=(5,5)) 
    plt.plot(fpr_Delta,tpr_Delta, color='r',linestyle='-',
              lw=lw, label='Delta ITH\nAUC=%.2f'%auc_score_Delta+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_Delta,auc_l_Delta, auc_h_Delta))
    plt.plot(fpr_BL,tpr_BL, color='b',linestyle='-',
              lw=lw, label='BL ITH\nAUC=%.2f'%auc_score_BL+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_BL, auc_l_BL, auc_h_BL))
    plt.plot(fpr_C1,tpr_C1, color='g',linestyle='-',
              lw=lw, label='C1 ITH\nAUC=%.2f'%auc_score_C1+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_C1, auc_l_C1, auc_h_C1))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    plt.legend(loc="lower right",edgecolor='k',title='Validation Cohort',fontsize=10,fancybox=False)
    plt.subplots_adjust(top=0.985,bottom=0.095,left=0.115,right=0.975,hspace=0,wspace=0)
    
    
    ## Comparision
    print('Training Cohort')
    P = delong_roc_test(Class_train, train_prob_Delta, train_prob_BL)
    print('Delta VS BL P:%.3f'%P[0][0])
    P = delong_roc_test(Class_train, train_prob_Delta, train_prob_C1)
    print('Delta VS C1 P:%.3f'%P[0][0])
    P = delong_roc_test(Class_train, train_prob_C1, train_prob_BL)
    print('C1 VS BL P:%.3f'%P[0][0])
    print('-----------------------------------------------')
    print('Testing Cohort')
    P = delong_roc_test(Class_test, test_prob_Delta, test_prob_BL)
    print('Delta VS BL P:%.3f'%P[0][0])
    P = delong_roc_test(Class_test, test_prob_Delta, test_prob_C1)
    print('Delta VS C1 P:%.3f'%P[0][0])
    P = delong_roc_test(Class_test, test_prob_C1, test_prob_BL)
    print('C1 VS BL P:%.3f'%P[0][0])
    print('-----------------------------------------------')
   