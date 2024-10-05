import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
from joblib import dump
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

def evaluate_metrics(y_true, y_pred_proba, threshold=0.5):
    # Convert predicted probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate Accuracy (Acc)
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate Sensitivity (Sn)
    sn = tp / (tp + fn)
    
    # Calculate Specificity (Sp)
    sp = tn / (tn + fp)
    
    # Calculate Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return acc, sn, sp, mcc, auc



# SVC Training and evaluation function
def train_and_evaluate_model_with_cv():
    # Load features and labels
    features = np.load('../Result/ST_sequence_extract_fea/train_esm2_fea.npy')
    labels = np.load('../Result/ST_sequence_extract_fea/train_labels.npy')

    # Print the shape of the loaded data
    print("Features shape:", features.shape)  # Output: (8616, 1280)
    print("Labels shape:", labels.shape)      # Output: (8616,)

    # '''
    # Create SVM model with probability output
    svm = SVC(kernel='rbf', probability=True, random_state=42)

    # Create five-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    metrics = {'acc': [], 'sn': [], 'sp': [], 'mcc': [], 'auc': []}

    for train_index, val_index in kf.split(features):
        # Create training and validation sets
        X_train, X_val = features[train_index], features[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        
        # Train SVM model
        svm.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = svm.predict_proba(X_val)[:, 1]
        
        # Calculate all evaluation metrics
        acc, sn, sp, mcc, auc = evaluate_metrics(y_val, y_pred_proba)
        metrics['acc'].append(acc)
        metrics['sn'].append(sn)
        metrics['sp'].append(sp)
        metrics['mcc'].append(mcc)
        metrics['auc'].append(auc)
    
        print(f"Fold: {fold}: Acc: {acc:.4f}, Sn: {sn:.4f}, Sp: {sp:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
        fold += 1
    
    # Print average metrics
    print(f"Average Metrics: Acc: {np.mean(metrics['acc']):.4f}, Sn: {np.mean(metrics['sn']):.4f}, Sp: {np.mean(metrics['sp']):.4f}, MCC: {np.mean(metrics['mcc']):.4f}, AUC: {np.mean(metrics['auc']):.4f}")

    # Re-train model using all data
    svm.fit(features, labels)

    # Save final model
    dump(svm, '../Result/ST_svm_final_model.joblib')
    print("Final SVM model saved to '../Result/ST_svm_final_model.joblib'")


# GBTree training and evaluation function
'''
def train_and_evaluate_model_with_cv():
    # Load features and labels
    features = np.load('../Result/ST_sequence_extract_fea/train_esm2_fea.npy')
    labels = np.load('../Result/ST_sequence_extract_fea/train_labels.npy')

    # Print the shape of the loaded data
    print("Features shape:", features.shape)  # Output: (8616, 1280)
    print("Labels shape:", labels.shape)      # Output: (8616,)

    # Create XGBClassifier model
    xgb_model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.2, 
        max_depth=7,
        subsample=0.8,
        colsample_bytree=1.0,
        gamma=0.2,
    )

    # Create five-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    metrics = {'acc': [], 'sn': [], 'sp': [], 'mcc': [], 'auc': []}

    for train_index, val_index in kf.split(features):
        # Create training and validation sets
        X_train, X_val = features[train_index], features[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        
        # Train XGBClassifier model
        xgb_model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
        
        # Calculate all evaluation metrics
        acc, sn, sp, mcc, auc = evaluate_metrics(y_val, y_pred_proba)
        metrics['acc'].append(acc)
        metrics['sn'].append(sn)
        metrics['sp'].append(sp)
        metrics['mcc'].append(mcc)
        metrics['auc'].append(auc)
    
        print(f"Fold: {fold}: Acc: {acc:.4f}, Sn: {sn:.4f}, Sp: {sp:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
        fold += 1
    
    # Print average metrics
    print(f"Average Metrics: Acc: {np.mean(metrics['acc']):.4f}, Sn: {np.mean(metrics['sn']):.4f}, Sp: {np.mean(metrics['sp']):.4f}, MCC: {np.mean(metrics['mcc']):.4f}, AUC: {np.mean(metrics['auc']):.4f}")

    # Re-train model using all data
    xgb_model.fit(features, labels)

    # Save final model
    dump(xgb_model, '../Result/ST_xgboostgbt_final_model.joblib')
    print("Final XGBoost model saved to '../Result/ST_xgboostgbt_final_model.joblib'")
'''
