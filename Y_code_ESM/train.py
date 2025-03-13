import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, EsmForSequenceClassification, RobertaForSequenceClassification
import numpy as np
import random
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold
import time

def train_and_evaluate_model_with_cv(train_data, train_label):

    # Device Setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set Random Seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../esm2_t33_650M_UR50D")

    # Define Metric Parameters
    accuracies = []
    val_sensitivity_list = []
    val_specificity_list = []
    val_mcc_list = []
    val_auc_list = []

    # Data Preprocessing
    encoded_texts = tokenizer(train_data, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(train_label, dtype=torch.float32).unsqueeze(1).to(device)

    # Define K-Fold Cross Validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Start Cross Validation
    for fold, (train_indices, val_indices) in enumerate(kf.split(input_ids)):
        print("##############")
        print(f"Fold {fold+1}:")
        print("##############")
        # Split Training and Validation Sets
        train_input_ids, train_attention_mask, train_labels = input_ids[train_indices], attention_mask[train_indices], labels[train_indices]
        val_input_ids, val_attention_mask, val_labels = input_ids[val_indices], attention_mask[val_indices], labels[val_indices]

        batch_size = 17

        # Create DataLoader
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize Variables and Model
        val_accuracy = 0
        val_sensitivity = 0
        val_specificity = 0
        val_predictions = []
        val_probabilities = []
        val_labels = []
        num_epochs = 4
        # total_steps = len(train_loader) * num_epochs
        total_steps = 5 * num_epochs
        print("train_loader length:", len(train_loader))
        # Model Parameter Initialization
        model = EsmForSequenceClassification.from_pretrained("../esm2_t33_650M_UR50D", num_labels=1).to(device)
        model.load_state_dict(torch.load('../Result/ST_esm2_t33_model.pth', map_location='cuda:1'))
        # Define Loss Function and Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Train the Model
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
                loss = outputs.loss
                logits = outputs.logits
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)
            print("  Average training loss: {0:.4f}".format(avg_loss))

            # Validate the Model for Each Epoch
            model.eval()
            # Tracking variables 
            for batch_input_ids, batch_attention_mask, batch_labels in val_loader:
                with torch.no_grad():
                    outputs = model(batch_input_ids, batch_attention_mask, 
                                    labels=batch_labels, return_dict=True)
                outputs = outputs.logits
                batch_preds = (outputs >= 0.5).squeeze().cpu().numpy()
                batch_probs = outputs.cpu().detach().numpy()
                val_predictions.extend(batch_preds.tolist())
                val_probabilities.extend(batch_probs.tolist())
                val_labels.extend(batch_labels.detach().cpu().numpy())

            # Calculate MCC, AUC
            val_mcc = matthews_corrcoef(val_labels, val_predictions)
            val_auc = roc_auc_score(val_labels, val_probabilities)

            # Calculate Sensitivity and Specificity
            TP = TN = FP = FN = 0
            for i in range(len(val_labels)):
                if val_predictions[i] == 1 and val_labels[i] == 1:
                    TP += 1
                elif val_predictions[i] == 0 and val_labels[i] == 0:
                    TN += 1
                elif val_predictions[i] == 1 and val_labels[i] == 0:
                    FP += 1
                else:
                    FN += 1
            val_sensitivity = TP / (TP + FN)
            val_specificity = TN / (TN + FP)
            val_accuracy = (TP + TN) / (TP + TN + FP + FN)
            print(f"Validation Accuracy: {val_accuracy:.4f} | Validation Sensitivity: {val_sensitivity:.4f} | Validation Specificity: {val_specificity:.4f} | Validation MCC: {val_mcc:.4f} | Validation AUC: {val_auc:.4f}")
            output_string = f"Validation Accuracy: {val_accuracy:.4f} | Validation Sensitivity: {val_sensitivity:.4f} | Validation Specificity: {val_specificity:.4f} | Validation MCC: {val_mcc:.4f} | Validation AUC: {val_auc:.4f}\n"

            with open("../Result/Y_output.txt", "a") as file:
                file.write(output_string)
            accuracies.append(val_accuracy)
            val_sensitivity_list.append(val_sensitivity)
            val_specificity_list.append(val_specificity)
            val_mcc_list.append(val_mcc)
            val_auc_list.append(val_auc)

    # Complete Model Training
    num_epochs = 4
    batch_size = 16
    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # total_steps = len(train_loader) * num_epochs
    total_steps = 5 * num_epochs
    model = EsmForSequenceClassification.from_pretrained("../esm2_t33_650M_UR50D", num_labels=1).to(device)
    model.load_state_dict(torch.load('../Result/ST_esm2_t33_modelbest.pth', map_location='cuda:1'))
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print("  Average training loss: {0:.4f}".format(avg_loss))

    model_save = model.state_dict()
    torch.save(model_save, '../Result/Y_esm2_t33_model.pth')

    # Print Cross Validation Results
    print("Cross Validation Results:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} | Average Sensitivity: {np.mean(val_sensitivity_list):.4f} | Average Specificity {np.mean(val_specificity_list):.4f} | Average MCC: {np.mean(val_mcc_list):.4f} | Average AUC: {np.mean(val_auc_list):.4f}")
    output_string = f"Cross Validation Results:\nAverage Accuracy: {np.mean(accuracies):.4f} | Average Sensitivity: {np.mean(val_sensitivity_list):.4f} | Average Specificity {np.mean(val_specificity_list):.4f} | Average MCC: {np.mean(val_mcc_list):.4f} | Average AUC: {np.mean(val_auc_list):.4f}\n"
    with open("../Result/Y_output.txt", "a") as file:
        file.write(output_string)
