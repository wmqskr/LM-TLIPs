import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, EsmForSequenceClassification, EsmModel
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import time
import torch.nn.functional as F
import numpy as np


def extract_fea(Data, label, m):
    # Device setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ESM2 feature extraction
    tokenizer = AutoTokenizer.from_pretrained("../esm2_t33_650M_UR50D")
    batch_size = 16
    esm2_fea_list = []
    labels_list = []
    n_samples = len(Data)
    n_batches = (n_samples + batch_size - 1) // batch_size  # Calculate total number of batches

    labels = torch.tensor(label, dtype=torch.float32).unsqueeze(1).to(device)  # Labels
    esm2model = EsmForSequenceClassification.from_pretrained("../esm2_t33_650M_UR50D",
                                                        output_hidden_states=True,
                                                        num_labels=1).to(device)
    print("esm2 word embedding")
    esm2model.load_state_dict(torch.load('../Result/ST_esm2_t33_model.pth', map_location='cuda:1'))
    esm2model.eval()

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_data = Data[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            encoded_texts = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_texts['input_ids'].to(device)
            attention_mask = encoded_texts['attention_mask'].to(device)
            outputs = esm2model(input_ids, attention_mask)
            batch_features = outputs.hidden_states[33]
            esm2_fea_list.append(batch_features)
            labels_list.append(batch_labels)

    esm2_fea = torch.cat(esm2_fea_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    esm2_fea = esm2_fea.permute(0, 2, 1)
    esm2_fea = F.max_pool1d(esm2_fea, kernel_size=35)
    esm2_fea = esm2_fea.squeeze(2)
    esm2_fea = esm2_fea.cpu().numpy()
    labels = labels.cpu().numpy()
    labels = labels.ravel()
    print("esm2 feature shape:", esm2_fea.shape)
    print("labels shape:", labels.shape)
    if m == 1:
        np.save("../Result/ST_sequence_extract_fea/train_esm2_fea.npy", esm2_fea)
        np.save("../Result/ST_sequence_extract_fea/train_labels.npy", labels)
    elif m == 2:
        np.save("../Result/ST_sequence_extract_fea/test_esm2_fea.npy", esm2_fea)
        np.save("../Result/ST_sequence_extract_fea/test_labels.npy", labels)
