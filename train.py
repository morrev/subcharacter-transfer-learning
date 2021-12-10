import importlib  
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

load_data = importlib.import_module("data.GDCE-SSA.src.util.load_data")

def livedoor_dataset():
    train_dataset = load_data('data/pickle/train_livedoor.pkl')['data']
    test_dataset = load_data('data/pickle/test_livedoor.pkl')['data']
    X_train, y_train = list(zip(*train_dataset))
    X_test, y_test = list(zip(*test_dataset))

    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.2, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
if __name__=='__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = livedoor_dataset
    print(X_train.shape, X_val.shape, X_test.shape, len(y_train), len(y_val), len(y_test))
