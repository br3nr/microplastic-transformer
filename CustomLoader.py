from numpy import dtype
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np

# CLASS: CustomDataset
# Purpose: Allows a custom dataset to be loaded into a DataLoader,
#          which is super convenient for PyTorch. 
class CustomDataset(Dataset):

    def __init__(self, feature, label):
        self.data = []
        for i in range(len(feature)):
            self.data.append([feature[i], label[i]])
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        feature, label = self.data[idx]
        feature_tensor = torch.tensor(feature,  dtype=torch.float32)
        label_tensor = torch.tensor(label)
        return feature_tensor, label_tensor