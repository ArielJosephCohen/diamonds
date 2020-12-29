import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import *
from xgboost import XGBClassifier, XGBRegressor
import datetime as dt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import eli5
from eli5.sklearn import PermutationImportance
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import time

def one_hot_encode(data, feature, keep_all = False):
    
    one_hot_cols = pd.get_dummies(data[feature])
    
    if keep_all == False:
        one_hot_cols=one_hot_cols.iloc[:,1:]
    new_data = pd.concat([data,one_hot_cols],axis=1)
    
    return new_data

def create_clf_dict(styles,X_tr,X_te,y_tr,y_te):
    clf_dict={}
    for i, model in enumerate(styles):
        clf_dict[i]=({'clf_name':str(model).split()[0],'clfs':model,
                    'X_train': X_tr,'X_test': X_te,'y_train': y_tr,'y_test': y_te})
    models_df = pd.DataFrame.from_dict(clf_dict, orient='index')
    
    return models_df

def gather_clf_data(models):
    
    clf_df = np.zeros((len(models), 26), dtype=object)
    
    for i,model in models.iterrows():
        
        clf_df[i,0] = model['clf_name']
        clf_df[i,1] = model['clfs']
                                       
        X_train = model['X_train']
        X_test = model['X_test']
        y_train = model['y_train']
        y_test = model['y_test']
                                       
        start_time = time.time()
                                       
        fitted = model['clfs'].fit(X_train, y_train)
        clf_df[i,2] = fitted
                                       
        stop_time = time.time()
        runtime = (stop_time - start_time)
        clf_df[i,3] = runtime
        
        pred_train = fitted.predict(X_train)
        pred_test = fitted.predict(X_test)
                                       
        clf_df[i,4] = pred_train
        clf_df[i,5] = pred_test
        
        y_score_train = fitted.predict_proba(X_train)
        y_score_test = fitted.predict_proba(X_test)
        clf_df[i,6] = y_score_train
        clf_df[i,7] = y_score_test
        
        clf_df[i,8], clf_df[i,9], thresholds_train = roc_curve(y_train, y_score_train[:,1])
        clf_df[i,10], clf_df[i,11], thresholds_test = roc_curve(y_test, y_score_test[:,1])
        
        clf_df[i,12] = precision_score(y_train,pred_train)
        clf_df[i,13] = precision_score(y_test, pred_test)
        clf_df[i,14] = recall_score(y_train,pred_train)
        clf_df[i,15] = recall_score(y_test, pred_test)
        clf_df[i,16] = f1_score(y_train,pred_train)
        clf_df[i,17] = f1_score(y_test, pred_test)
        clf_df[i,18] = accuracy_score(y_train,pred_train)
        clf_df[i,19] = accuracy_score(y_test, pred_test)
        clf_df[i,20] = roc_auc_score(y_train,pred_train)
        clf_df[i,21] = roc_auc_score(y_test, pred_test)                            
                                       
        clf_df[i,22] = X_train
        clf_df[i,23] = X_test
        clf_df[i,24] = y_train
        clf_df[i,25] = y_test 
                            
    columns = ['Classifier',
               'Model',
               'Fitted Model',
               'Runtime',
               'Train Preds',
               'Test Preds',
               'Train y-Score',
               'Test y-Score',
               'Train FPR',
               'Train TPR',
               'Test FPR',
               'Test TPR',
               'Train Precision',
               'Test Precision',
               'Train Recall',
               'Test Recall',
               'Train F1',
               'Test F1',
               'Train Accuracy',
               'Test Accuracy',
               'Train ROC AUC',
               'Test ROC AUC',
               'X_train',
               'X_test',
               'y_train',
               'y_test'
              ]
        
    clf_df = pd.DataFrame(clf_df, columns=columns)
    
    return clf_df
                                       
def normalized_cm(clf_df):
                                       
    for i,clf in clf_df.iterrows():
        
        classifier = clf['Classifier']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
        fig.suptitle(f"{classifier}:") 

        plot_confusion_matrix(clf['Model'],
                                    clf['X_train'],
                                    clf['y_train'],
                                    cmap='winter',
                                    normalize='true',
                                    ax=ax1)
        ax1.set_title('Train')

        plot_confusion_matrix(clf['Model'], 
                                      clf['X_test'], 
                                      clf['y_test'], 
                                      cmap='winter',
                                      normalize='true',
                                      ax=ax2)
        ax2.set_title('Test')
                                       
        plt.tight_layout()
        plt.show()
        
def total_cm(clf_df):
    
    for i, clf in clf_df.iterrows():
        
        classifier=clf['Classifier']
        fig.suptitle(f'{Classifier}')
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,8))
        sns.heatmap(confusion_matrix(y_train,pred_train),annot=True,cmap='winter',ax=ax1)
        ax1.set_title('Train')
        sns.heatmap(confusion_matrix(y_test,pred_test),annot=True,cmap='winter',ax=ax2)
        ax2.set_title('Test')
        plt.tight_layout()
        plt.show()
        
def clf_reports(clf_df):
    
    print('Train')
    print(classification_report(clf['y_train'],clf['pred_train']))
    
    print('-'*20)
    
    print('Test')
    print(classification_report(clf['y_test'],clf['pred_test']))
                                       
                                       
def plot_roc(clf_df):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
    fig.suptitle('ROC Curves', fontsize=24)
    
    for i, clf in clf_df.iterrows():
        
        classifier = clf['Classifier']
        
        ax1.plot(clf['Train FPR'],clf['Train TPR'],linestyle='-',
        lw=3,label=(str(classifier) + ' - AUC:' + str("%0.2f" %    clf['Train ROC AUC'])))
        
        ax1.set_title('Train')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive rate')
        ax1.plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--')
        ax1.set_xlim([-0.05, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.legend(loc='lower right')
                        
        ax2.plot(clf['Test FPR'], 
                 clf['Test TPR'], 
                 linestyle='-', 
                 lw=3, 
                 label=(str(classifier) + ' - AUC:' + str("%0.2f" % clf['Test ROC AUC']))
                )

        ax2.set_title('Test')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive rate')          
        ax2.plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--')
        ax2.set_xlim([-0.05, 1.0])
        ax2.set_ylim([0.0, 1.05])              
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
    return

def feat_rank(clf_df):
    for i, clf in clf_df.iterrows():
                       
        classifier = clf['Classifier']
        p = classifier.fit(X_train, y_train)
        perm = PermutationImportance(p).fit(X_test, y_test)
        eli5.show_weights(perm)

    return 
                                       
def grid_search(params, clf_df, cv=3, scoring='recall'):
                                       
    for i, clf in clf_df.iterrows():
        print(clf['Classifier'])
        print(f"\nParameter Grid: {params[i]} \n")
        clf_grid_search = GridSearchCV(clf['Model'],
                                  param_grid=params[i],
                                  cv=cv,
                                  scoring=scoring)
        clf_grid_search.fit(clf['X_train'], clf['y_train'])
        # print out best parameters rounded to the second decimal place with the percent added to display as percent   
        # and not decimal
        # also, we add the decimal before the 2% to allow for rounding
        print(f"Training Accuracy: {clf_grid_search.best_score_ :.2%} \n")
        print(f"Optimal Parameters: {clf_grid_search.best_params_} \n\n")
        
