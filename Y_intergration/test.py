import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, EsmForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef, auc, roc_curve
from torch.utils.data import TensorDataset
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt

def test_model(test_data, test_label):
    # Device setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

#--------------------------------ESM-2----------------------------------------#
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("../esm2_t33_650M_UR50D")
    model = EsmForSequenceClassification.from_pretrained("../esm2_t33_650M_UR50D",
                                                        num_labels=1).to(device)
    
    # Load the best saved model
    model.load_state_dict(torch.load('../Result/Y_esm2_t33_model.pth', map_location='cuda:0'))
    model.eval()

    # Data preprocessing
    encoded_texts = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(test_label, dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoader
    test_dataset = TensorDataset(input_ids, attention_mask, labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Inference on test data
    all_preds = []
    all_probs = []
    all_labels = []
    for batch_input_ids, batch_attention_mask, batch_labels in test_loader:
        with torch.no_grad():
            outputs = model(batch_input_ids, batch_attention_mask, 
                            labels=batch_labels, return_dict=True)
        outputs = outputs.logits
        batch_preds = (outputs >= 0.5).squeeze().cpu().numpy()
        batch_probs = outputs.cpu().detach().numpy()
        all_preds.extend(batch_preds.tolist())
        all_probs.extend(batch_probs.tolist())
        all_labels.extend(batch_labels.detach().cpu().numpy())
    esm_preds = all_preds
    esm_probs = all_probs
#--------------------------------ESM-2----------------------------------------#

#--------------------------------machine learning-----------------------------#
    # Load test set features and labels
    test_features = np.load('../Result/Y_sequence_extract_fea/test_esm2_fea.npy')
    test_labels = np.load('../Result/Y_sequence_extract_fea/test_labels.npy')

    # Load trained XGBoost model
    xgb_model = load('../Result/YbasedST_xgboostgbt_final_model.joblib')
    # Predict using the model
    xgb_pred_proba = xgb_model.predict_proba(test_features)[:, 1]

#--------------------------------machine learning-----------------------------#

    ensemble_preds = []
    ensemble_probs = []
    for i in range(len(esm_probs)):
        ensemble_probs.append((esm_probs[i] + xgb_pred_proba[i]) / 2)

        if ensemble_probs[i] >= 0.5:
            ensemble_preds.append(1)
        else:
            ensemble_preds.append(0)

    # Calculate metrics
    mcc_value = matthews_corrcoef(all_labels, ensemble_preds)
    auc_value = roc_auc_score(all_labels, ensemble_probs)
    TP = TN = FP = FN = 0
    for i in range(len(all_labels)):
        if ensemble_preds[i] == 1 and all_labels[i] == 1:
            TP += 1
        elif ensemble_preds[i] == 0 and all_labels[i] == 0:
            TN += 1 
        elif ensemble_preds[i] == 1 and all_labels[i] == 0:
            FP += 1
        else:
            FN += 1
    acc = (TP + TN) / (TP + TN + FP + FN)
    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    print(f"Test Acc: {acc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}, MCC: {mcc_value:.4f}, AUC: {auc_value:.4f}")

