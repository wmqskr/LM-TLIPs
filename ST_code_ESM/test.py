import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, EsmForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from torch.utils.data import TensorDataset

def test_model(test_data, test_label):
    # Device setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("../esm2_t33_650M_UR50D")
    model = EsmForSequenceClassification.from_pretrained("../esm2_t33_650M_UR50D",
                                                        num_labels=1).to(device)
    
    # Load the best saved model
    model.load_state_dict(torch.load('../Result/ST_esm2_t33_model.pth', map_location='cuda:1'))
    model.eval()

    # Data preprocessing
    encoded_texts = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(test_label, dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoader
    test_dataset = TensorDataset(input_ids, attention_mask, labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Inference on the test data
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

    # Calculate metrics
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    TP = TN = FP = FN = 0
    for i in range(len(all_labels)):
        if all_preds[i] == 1 and all_labels[i] == 1:
            TP += 1
        elif all_preds[i] == 0 and all_labels[i] == 0:
            TN += 1 
        elif all_preds[i] == 1 and all_labels[i] == 0:
            FP += 1
        else:
            FN += 1
    acc = (TP + TN) / (TP + TN + FP + FN)
    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    print(f"Test Acc: {acc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
