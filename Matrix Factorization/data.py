import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd
import numpy as np
np.random.seed(42)


def train_test_split(split_ratio):
    dataset = pd.read_csv('../Datasets/ml-100k/u.data', header = None, sep = '\t' )
    #print(dataset.head())
    print(f'total entries {len(dataset)} ')

    print(f'unique users {len(dataset[0].unique())}')
    print(f'unique items {len(dataset[1].unique())}')
    print(f'unique values {len(dataset[2].unique())}')

    '''
    Here we are doing random split because we are not looking for
    sequential behaviour

    '''

    # print(min(dataset[0].unique())) # 1 
    # print(max(dataset[0].unique())) # 943


    # print(min(dataset[1].unique())) # 1
    # print(max(dataset[1].unique())) # 1682

    # we have to convert it to starting from zero

    '''
    new_list_196 = np.array(dataset[dataset[0] == 196 ][1])
    new_list_186 = np.array(dataset[dataset[0] == 186][1])
    print(new_list_196.shape)
    print(new_list_186.shape)
    '''
    item_dict = {}
    user_dict = {}

    user_id = 0
    item_id = 0
    for index , row_data in dataset.iterrows():
        if(row_data[0] not in user_dict):
            user_dict[ row_data[0] ] = user_id
            user_id += 1
        if(row_data[1] not in item_dict):
            item_dict[ row_data[1] ] = item_id
            item_id += 1
    '''
    print(user_id, len(user_dict))
    print(item_id, len(item_dict))
    '''

    dataset[0] = dataset[0].replace(user_dict)
    dataset[1] = dataset[1].replace(item_dict)

    '''
    print(dataset.head())

    new_list_196 = np.array(dataset[dataset[0] == user_dict[196] ][1])
    new_list_186 = np.array(dataset[dataset[0] == user_dict[186]][1])
    print(new_list_196.shape)
    print(new_list_186.shape)
    '''

            
    ###################
    #Now spllit the datset into train_test set



    mask = np.random.uniform(0, 1 , len(dataset) ) < 1 - split_ratio
    neg_mask = np.array([not x for x in mask])
    train , test = dataset.iloc[mask], dataset.iloc[neg_mask]
    return np.array(train), np.array(test), user_id, item_id


#train = np.array(train)
#test = np.array(test)


class my_dataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data[:,0], dtype = torch.long)
        self.items = torch.tensor(data[:,1], dtype = torch.long)
        self.ratings = torch.tensor(data[:,2], dtype = torch.float)
        self.len = data.shape[0]
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.ratings[index]
    
    
    def __len__(self):
        return self.len

