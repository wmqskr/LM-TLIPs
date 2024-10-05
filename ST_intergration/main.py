import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from test import test_model


# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

    
# Extract sequences
def remove_name(data):
    data_new = []
    for i in range(1, len(data), 2):
        data_new.append(data[i])
    return data_new


# Read data
with open("../data/ST-train.fa") as f:
    ST_train = f.readlines()
    ST_train = [s.strip() for s in ST_train]
with open("../data/ST-test.fa") as f:
    ST_test = f.readlines()
    ST_test = [s.strip() for s in ST_test]
print(len(ST_train), len(ST_test))
print("Data reading completed")
print("———————————————————————————————————————————————————")

ST_train_x = remove_name(ST_train)
ST_test_x = remove_name(ST_test)

print(len(ST_train_x), len(ST_train_x[0]))
print(len(ST_test_x), len(ST_test_x[0]))
print("Sequence extraction completed")
print("———————————————————————————————————————————————————")


# Define labels
ST_train_y = np.concatenate([np.ones((int(len(ST_train_x) / 2),)), np.zeros((int(len(ST_train_x) / 2),))], axis=0)  # Vertical concatenation
ST_test_y = np.concatenate([np.ones((int(len(ST_test_x) / 2),)), np.zeros((int(len(ST_test_x) / 2),))], axis=0)
print(ST_train_y.shape, ST_test_y.shape)
print("Label definition completed")
print("———————————————————————————————————————————————————")


# Call test function
test_model(ST_test_x, ST_test_y)
print("Testing completed")
