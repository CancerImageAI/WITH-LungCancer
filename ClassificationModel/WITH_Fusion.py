# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:33:02 2024

@author: DELL
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc,confusion_matrix
import matplotlib.pyplot as plt

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
    

if __name__ == '__main__':
    font = {'family' : 'Times New Roman',
 			'weight' : 'normal',
 			'size'   : 12,}
    plt.rc('font', **font)

    ITH_Train = pd.read_csv('../result/ITHtumor_Result_Training.csv')
    ID_ITH_Train = list(np.array(ITH_Train['ID']))
    ITH_Score_Train = np.array(ITH_Train['Fusion_Score'])
    Class_Train = np.array(ITH_Train['Class'])
    
    Train_Result = {}
    Train_Result['ID'] = ID_ITH_Train
    Train_Result['Class'] = Class_Train
    Train_Result['ITH'] = ITH_Score_Train

    
    ITH_Test = pd.read_csv('../result/ITHtumor_Result_Testing.csv')
    ID_ITH_Test = list(np.array(ITH_Test['ID']))
    ITH_Score_Test = np.array(ITH_Test['Fusion_Score'])
    Class_Test = np.array(ITH_Test['Class'])
    
    Test_Result = {}
    Test_Result['ID'] = ID_ITH_Test
    Test_Result['Class'] = Class_Test
    Test_Result['ITH'] = ITH_Score_Test

    
    ITH_FUSCC = pd.read_csv('../result/ITHtumor_Result_FUSCC.csv')
    ID_ITH_FUSCC = list(np.array(ITH_FUSCC['ID']))
    ITH_Score_FUSCC = np.array(ITH_FUSCC['Fusion_Score'])
    Class_FUSCC = np.array(ITH_FUSCC['Class'])
    
    FUSCC_Result = {}
    FUSCC_Result['ID'] = ID_ITH_FUSCC
    FUSCC_Result['Class'] = Class_FUSCC
    FUSCC_Result['ITH'] = ITH_Score_FUSCC

    
    ITH_FUZSH = pd.read_csv('../result/ITHtumor_Result_FUZSH.csv')
    ID_ITH_FUZSH = list(np.array(ITH_FUZSH['ID']))
    ITH_Score_FUZSH = np.array(ITH_FUZSH['Fusion_Score'])
    Class_FUZSH = np.array(ITH_FUZSH['Class'])
    
    FUZSH_Result = {}
    FUZSH_Result['ID'] = ID_ITH_FUZSH
    FUZSH_Result['Class'] = Class_FUZSH
    FUZSH_Result['ITH'] = ITH_Score_FUZSH

    
    WTH_Train = pd.read_csv('../result/Wholetumor_Result_Training.csv')
    ID_WTH_Train = list(np.array(WTH_Train['ID']))
    WTH_Score_Train = np.array(WTH_Train['Fusion_Score'])
    ind_train = [ID_WTH_Train.index(i) for i in ID_ITH_Train]
    WTH_Score_Train = WTH_Score_Train[ind_train]
    
    Train_Result['WTH'] = WTH_Score_Train
    
    WTH_Test = pd.read_csv('../result/Wholetumor_Result_Testing.csv')
    ID_WTH_Test = list(np.array(WTH_Test['ID']))
    WTH_Score_Test = np.array(WTH_Test['Fusion_Score'])
    ind_test = [ID_WTH_Test.index(i) for i in ID_ITH_Test]
    WTH_Score_Test = WTH_Score_Test[ind_test]
    Test_Result['WTH'] = WTH_Score_Test
    
    WTH_FUSCC = pd.read_csv('../result/Wholetumor_Result_FUSCC.csv')
    ID_WTH_FUSCC = list(np.array(WTH_FUSCC['ID']))
    WTH_Score_FUSCC = np.array(WTH_FUSCC['Fusion_Score'])
    ind_FUSCC = [ID_WTH_FUSCC.index(i) for i in ID_ITH_FUSCC]
    WTH_Score_FUSCC = WTH_Score_FUSCC[ind_FUSCC]
    FUSCC_Result['WTH'] = WTH_Score_FUSCC
    
    WTH_FUZSH = pd.read_csv('../result/Wholetumor_Result_FUZSH.csv')
    ID_WTH_FUZSH = list(np.array(WTH_FUZSH['ID']))
    WTH_Score_FUZSH = np.array(WTH_FUZSH['Fusion_Score'])
    ind_FUZSH = [ID_WTH_FUZSH.index(i) for i in ID_ITH_FUZSH]
    WTH_Score_FUZSH = WTH_Score_FUZSH[ind_FUZSH]
    FUZSH_Result['WTH'] = WTH_Score_FUZSH
    
    scales = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for scale in scales: 
        train_prob_fusion = scale*np.array(ITH_Score_Train)+(1-scale)*np.array(WTH_Score_Train)
        train_auc_fusion = roc_auc_score(np.array(Class_Train),ITH_Score_Train)
        auc_fl_fusion, auc_fh_fusion, auc_fstd_fusion = confindence_interval_compute(np.array(train_prob_fusion), Class_Train)
        print('training Dataset Fusion Scale',scale,'AUC:%.3f'%train_auc_fusion,'+/-%.2f'%auc_fstd_fusion,
              '  95% CI:[','%.2f,'%auc_fl_fusion,'%.2f'%auc_fh_fusion,']')      
    train_Fusion_fusion = np.zeros([len(ITH_Score_Train),2])
    train_Fusion_fusion[:,0] = np.array(ITH_Score_Train)
    train_Fusion_fusion[:,1] = np.array(WTH_Score_Train)
    train_Fusion_fusion_min = train_Fusion_fusion.min(1)
    train_Fusion_fusion_max = train_Fusion_fusion.max(1)

    train_auc_fusion_min = roc_auc_score(np.array(Class_Train),train_Fusion_fusion_min)
    auc_fl_fusion_min, auc_fh_fusion_min, auc_fstd_fusion_min = confindence_interval_compute(np.array(train_Fusion_fusion_min), Class_Train)
    print('Min Fusion AUC:%.3f'%train_auc_fusion_min,'+/-%.2f'%auc_fstd_fusion_min,'  95% CI:[','%.2f,'%auc_fl_fusion_min,'%.2f'%auc_fh_fusion_min,']')
    
    train_auc_fusion_max = roc_auc_score(np.array(Class_Train),train_Fusion_fusion_max)
    auc_fl_fusion_max, auc_fh_fusion_max,auc_fstd_fusion_max = confindence_interval_compute(np.array(train_Fusion_fusion_max), Class_Train)
    print('Max Fusion AUC:%.3f'%train_auc_fusion_max,'+/-%.2f'%auc_fstd_fusion_max,'  95% CI:[','%.2f,'%auc_fl_fusion_max,'%.2f'%auc_fh_fusion_max,']')
    print('----------------------------------------------') 
    prediction_score(np.array(Class_Train), train_prob_fusion>0.5)
    Train_Result['Fusion'] = train_prob_fusion
    
    for scale in scales: 
        test_prob_fusion = scale*np.array(ITH_Score_Test)+(1-scale)*np.array(WTH_Score_Test)
        test_auc_fusion = roc_auc_score(np.array(Class_Test),test_prob_fusion)
        auc_fl_fusion, auc_fh_fusion, auc_fstd_fusion = confindence_interval_compute(np.array(test_prob_fusion), Class_Test)
        print('testing Dataset Fusion Scale',scale,'AUC:%.3f'%test_auc_fusion,'+/-%.2f'%auc_fstd_fusion,
              '  95% CI:[','%.2f,'%auc_fl_fusion,'%.2f'%auc_fh_fusion,']')      
    test_Fusion_fusion = np.zeros([len(ITH_Score_Test),2])
    test_Fusion_fusion[:,0] = np.array(ITH_Score_Test)
    test_Fusion_fusion[:,1] = np.array(WTH_Score_Test)
    test_Fusion_fusion_min = test_Fusion_fusion.min(1)
    test_Fusion_fusion_max = test_Fusion_fusion.max(1)

    test_auc_fusion_min = roc_auc_score(np.array(Class_Test),test_Fusion_fusion_min)
    auc_fl_fusion_min, auc_fh_fusion_min, auc_fstd_fusion_min = confindence_interval_compute(np.array(test_Fusion_fusion_min), Class_Test)
    print('Min Fusion AUC:%.3f'%test_auc_fusion_min,'+/-%.2f'%auc_fstd_fusion_min,'  95% CI:[','%.2f,'%auc_fl_fusion_min,'%.2f'%auc_fh_fusion_min,']')
    
    test_auc_fusion_max = roc_auc_score(np.array(Class_Test),test_Fusion_fusion_max)
    auc_fl_fusion_max, auc_fh_fusion_max,auc_fstd_fusion_max = confindence_interval_compute(np.array(test_Fusion_fusion_max), Class_Test)
    print('Max Fusion AUC:%.3f'%test_auc_fusion_max,'+/-%.2f'%auc_fstd_fusion_max,'  95% CI:[','%.2f,'%auc_fl_fusion_max,'%.2f'%auc_fh_fusion_max,']')
    print('----------------------------------------------') 
    
    Test_Result['Fusion'] = test_Fusion_fusion_min
    
    for scale in scales: 
        FUSCC_prob_fusion = scale*np.array(ITH_Score_FUSCC)+(1-scale)*np.array(WTH_Score_FUSCC)
        FUSCC_auc_fusion = roc_auc_score(np.array(Class_FUSCC),FUSCC_prob_fusion)
        auc_fl_fusion, auc_fh_fusion, auc_fstd_fusion = confindence_interval_compute(np.array(FUSCC_prob_fusion), Class_FUSCC)
        print('FUSCCing Dataset Fusion Scale',scale,'AUC:%.3f'%FUSCC_auc_fusion,'+/-%.2f'%auc_fstd_fusion,
              '  95% CI:[','%.2f,'%auc_fl_fusion,'%.2f'%auc_fh_fusion,']')      
    FUSCC_Fusion_fusion = np.zeros([len(ITH_Score_FUSCC),2])
    FUSCC_Fusion_fusion[:,0] = np.array(ITH_Score_FUSCC)
    FUSCC_Fusion_fusion[:,1] = np.array(WTH_Score_FUSCC)
    FUSCC_Fusion_fusion_min = FUSCC_Fusion_fusion.min(1)
    FUSCC_Fusion_fusion_max = FUSCC_Fusion_fusion.max(1)

    FUSCC_auc_fusion_min = roc_auc_score(np.array(Class_FUSCC),FUSCC_Fusion_fusion_min)
    auc_fl_fusion_min, auc_fh_fusion_min, auc_fstd_fusion_min = confindence_interval_compute(np.array(FUSCC_Fusion_fusion_min), Class_FUSCC)
    print('Min Fusion AUC:%.3f'%FUSCC_auc_fusion_min,'+/-%.2f'%auc_fstd_fusion_min,'  95% CI:[','%.2f,'%auc_fl_fusion_min,'%.2f'%auc_fh_fusion_min,']')
    
    FUSCC_auc_fusion_max = roc_auc_score(np.array(Class_FUSCC),FUSCC_Fusion_fusion_max)
    auc_fl_fusion_max, auc_fh_fusion_max,auc_fstd_fusion_max = confindence_interval_compute(np.array(FUSCC_Fusion_fusion_max), Class_FUSCC)
    print('Max Fusion AUC:%.3f'%FUSCC_auc_fusion_max,'+/-%.2f'%auc_fstd_fusion_max,'  95% CI:[','%.2f,'%auc_fl_fusion_max,'%.2f'%auc_fh_fusion_max,']')
    print('----------------------------------------------') 
    
    FUSCC_Result['Fusion'] = FUSCC_Fusion_fusion_max
    
    for scale in scales: 
        FUZSH_prob_fusion = scale*np.array(ITH_Score_FUZSH)+(1-scale)*np.array(WTH_Score_FUZSH)
        FUZSH_auc_fusion = roc_auc_score(np.array(Class_FUZSH),FUZSH_prob_fusion)
        auc_fl_fusion, auc_fh_fusion, auc_fstd_fusion = confindence_interval_compute(np.array(FUZSH_prob_fusion), Class_FUZSH)
        print('FUZSHing Dataset Fusion Scale',scale,'AUC:%.3f'%FUZSH_auc_fusion,'+/-%.2f'%auc_fstd_fusion,
              '  95% CI:[','%.2f,'%auc_fl_fusion,'%.2f'%auc_fh_fusion,']')      
    FUZSH_Fusion_fusion = np.zeros([len(ITH_Score_FUZSH),2])
    FUZSH_Fusion_fusion[:,0] = np.array(ITH_Score_FUZSH)
    FUZSH_Fusion_fusion[:,1] = np.array(WTH_Score_FUZSH)
    FUZSH_Fusion_fusion_min = FUZSH_Fusion_fusion.min(1)
    FUZSH_Fusion_fusion_max = FUZSH_Fusion_fusion.max(1)

    FUZSH_auc_fusion_min = roc_auc_score(np.array(Class_FUZSH),FUZSH_Fusion_fusion_min)
    auc_fl_fusion_min, auc_fh_fusion_min, auc_fstd_fusion_min = confindence_interval_compute(np.array(FUZSH_Fusion_fusion_min), Class_FUZSH)
    print('Min Fusion AUC:%.3f'%FUZSH_auc_fusion_min,'+/-%.2f'%auc_fstd_fusion_min,'  95% CI:[','%.2f,'%auc_fl_fusion_min,'%.2f'%auc_fh_fusion_min,']')
    
    FUZSH_auc_fusion_max = roc_auc_score(np.array(Class_FUZSH),FUZSH_Fusion_fusion_max)
    auc_fl_fusion_max, auc_fh_fusion_max,auc_fstd_fusion_max = confindence_interval_compute(np.array(FUZSH_Fusion_fusion_max), Class_FUZSH)
    print('Max Fusion AUC:%.3f'%FUZSH_auc_fusion_max,'+/-%.2f'%auc_fstd_fusion_max,'  95% CI:[','%.2f,'%auc_fl_fusion_max,'%.2f'%auc_fh_fusion_max,']')
    print('----------------------------------------------') 
    
    FUZSH_Result['Fusion'] = FUZSH_Fusion_fusion_max
    
    train_df = pd.DataFrame(Train_Result)
    train_df.to_csv(r'../result/WITHtumor_Result_Training.csv')
    
    test_df = pd.DataFrame(Test_Result)
    test_df.to_csv(r'../result/WITHtumor_Result_Testing.csv')
    
    FUSCC_df = pd.DataFrame(FUSCC_Result)
    FUSCC_df.to_csv(r'../result/WITHtumor_Result_FUSCC.csv')
    
    FUZSH_df = pd.DataFrame(FUZSH_Result)
    FUZSH_df.to_csv(r'../result/WITHtumor_Result_FUZSH.csv')