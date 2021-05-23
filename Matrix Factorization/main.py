import torch
import torch.nn as nn
import data
from torch.utils.data import DataLoader



######################       HYPERPARAMETERS       #############
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 5
n_epoch = 10
lr = 0.01
dropout = 0.5





train, test = data.train_test_split(0.1)
print(f'training set contains {train.shape[0]}')
print(f'test set contains {test.shape[0]}')




train_dataset = data.my_dataset(train)
test_dataset = data.my_dataset(test)

train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size)

'''
# to cheeck correctness of data
print(test[:5])
for x, y , z in test_loader:
    print(x,y,z)
    break
'''


