import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from train import train_and_evaluate_model_with_cv
from test import test_model
import random


# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

    
# Extract sequences
def remove_name(data):
    data_new = []
    for i in range(1, len(data), 2):
        data_new.append(data[i])
    return data_new


# Read data
with open("../data/Y-train.fa") as f:
    Y_train = f.readlines()
    Y_train = [s.strip() for s in Y_train]
with open("../data/Y-test.fa") as f:
    Y_test = f.readlines()
    Y_test = [s.strip() for s in Y_test]
print(len(Y_train), len(Y_test))
print("Data reading completed")
print("———————————————————————————————————————————————————")

Y_train_x = remove_name(Y_train)
Y_test_x = remove_name(Y_test)

print(len(Y_train_x), len(Y_train_x[0]))
print(len(Y_test_x), len(Y_test_x[0]))
print("Sequence extraction completed")
print("———————————————————————————————————————————————————")


# Define labels
Y_train_y = np.concatenate([np.ones((int(len(Y_train_x) / 2),)), np.zeros((int(len(Y_train_x) / 2),))], axis=0)  # Vertical stacking
Y_test_y = np.concatenate([np.ones((int(len(Y_test_x) / 2),)), np.zeros((int(len(Y_test_x) / 2),))], axis=0)
print(Y_train_y.shape, Y_test_y.shape)
print("Label definition completed")
print("———————————————————————————————————————————————————")

n = int(input("1:training 2:testing:"))
if n == 1:
    # Call training function
    print("ST_site")
    print("Model: ESM2")
    train_and_evaluate_model_with_cv(Y_train_x, Y_train_y)
    print("Model: ESM2")
    print("Training completed")

else:
    # Call testing function
    test_model(Y_test_x, Y_test_y)
    print("Testing completed")
