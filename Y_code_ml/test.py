import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from joblib import dump, load
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def evaluate_metrics(y_true, y_pred_proba, threshold=0.5):
    # Convert predicted probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_proba)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate accuracy (Acc)
    acc = accuracy_score(y_true, y_pred)

    # Calculate sensitivity (Sn)
    sn = tp / (tp + fn)

    # Calculate specificity (Sp)
    sp = tn / (tn + fp)

    # Calculate Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, sn, sp, mcc, auc


# GBTree test function
'''
def test_model():
    # Load test set features and labels
    test_features = np.load('../Result/Y_sequence_extract_fea/test_esm2_fea.npy')
    test_labels = np.load('../Result/Y_sequence_extract_fea/test_labels.npy')

    # Load trained XGBoost model
    xgb_model = load('../Result/YbasedST_xgboostgbt_final_model.joblib')

    # Use model to make predictions
    y_pred_proba = xgb_model.predict_proba(test_features)[:, 1]

    # Calculate all evaluation metrics
    acc, sn, sp, mcc, auc = evaluate_metrics(test_labels, y_pred_proba)

    # Print evaluation results
    print(f"Independent Test Set Metrics: Acc: {acc:.4f}, Sn: {sn:.4f}, Sp: {sp:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
'''
