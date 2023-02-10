import numpy as np
import pandas as pd
from CustomLoader import CustomDataset
from torch.utils.data import Dataset, DataLoader
from src.augmentation import *

LABEL_DICT = {'PP': 0, 'PE':1}

def reshape_data(data_x, data_y):
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], 1))
    data_y = data_y.reshape((data_y.shape[0]))
    
    # Shuffle the array
    idx = np.random.permutation(len(data_x))
    data_x = data_x[idx]
    data_y = data_y[idx]
    
    # Make the labels not strings
    data_y[data_y == -1] = 0 
    data_y = np.array([LABEL_DICT[zi] for zi in data_y])
    return data_x, data_y


def load_data(path_x, path_y, csv=True):
    if(csv):
        data_x = pd.read_csv(path_x, header=0, dtype=float).to_numpy()
    else:
        data_x = pd.read_csv(path_x, header=0, dtype=float, sep='\t').to_numpy()

    data_y = pd.read_csv(path_y, header=0).to_numpy()
    return data_x, data_y


def reshape_test_data(test_x, test_y):
    test_x  = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    test_y  = test_y.reshape((test_y.shape[0]))
    test_y[test_y == -1] == 0
    test_y = np.array([LABEL_DICT[zi] for zi in test_y])
    test_x = np.asarray(test_x).astype('float32')
    return test_x, test_y


def difference(dataset, interval=1):
        diff = []
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i-interval]
            diff.append(value)
        return diff


def get_norder(dataset, order, diff_val):
    differences = []
    for i in range(dataset.shape[0]):
        diff = dataset[i]
        for i in range(order):
            diff = difference(diff, interval=diff_val)
        differences.append(diff)
    differences = np.asarray(differences)
    return differences


def get_dataloader(path_x, path_y, bs, csv=True):
    data_x, data_y = load_data(path_x, path_y)
    data_x = get_norder(data_x, 4, 1)
    data_x, data_y = reshape_data(data_x, data_y)
    shape = data_x.shape[1]
    data = CustomDataset(data_x, data_y)
    return DataLoader(dataset=data, batch_size=bs, shuffle=True), shape


def get_test_dataloader(path_x, path_y, bs, csv=True):
    test_x, test_y = load_data(path_x, path_y, csv=csv)
    test_x = get_norder(test_x, 4, 1)
    test_x, test_y = reshape_test_data(test_x, test_y)
    test = CustomDataset(test_x, test_y)
    return DataLoader(dataset=test, batch_size=bs, shuffle=False)