import torch
import torch.nn as nn

import model
import data
import trainer
import pandas as pd
import numpy as np
######################       HYPERPARAMETERS       #############
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 100
n_epoch = 20
lr = 0.01
dropout = 0.5
n_factor = 30
weight_decay = 0.0001

logs = pd.DataFrame(columns=('batch_size', 'lr', 'n_factor', 'weight_decay', 'train_loss', 'test_loss'))
logs_list = []




train, test, n_users, n_items = data.train_test_split(0.1)
print(f'training set contains {train.shape[0]}')
print(f'test set contains {test.shape[0]}')
train_dataset = data.my_dataset(train)
test_dataset = data.my_dataset(test)

n_factors = [10,20,30, 40, 50, 50, 70, 80]
batch_sizes = [32, 64, 128, 256, 512, 1024]
lrs = [0.00001,0.0001,0.001,0.1]
min_loss = 10
for n_factor in n_factors:
    for batch_size in batch_sizes:

        for lr in lrs:
            our_model_trainer = trainer.trainer(train_dataset = train_dataset,
                                        test_dataset = test_dataset,
                                        batch_size = batch_size,
                                        lr = lr,
                                        n_epoch = n_epoch,
                                        n_factor = n_factor,
                                        n_user = n_users,
                                        n_item = n_items,
                                        weight_decay = weight_decay)

            best_params, train_rmse, test_rmse = our_model_trainer.train()

            logs_list.append(best_params['batch_size'])
            logs_list.append(best_params['lr'])
            logs_list.append(best_params['n_factor'])
            logs_list.append(weight_decay)
            logs_list.append(train_rmse)
            logs_list.append(test_rmse)
            df_length = logs.shape[0]
            #print(df_length)
            logs.loc[df_length] = logs_list
            logs_list = []
            
            if(best_params['min_loss'] < min_loss):
                obtained_dict = best_params
                min_loss = best_params['min_loss']
                best_lr = lr
                best_n_factor = n_factor
                best_batch_size = batch_size
            print()
            print()

print(f'best  {best_batch_size} and min loss {min_loss} {best_lr}, {best_n_factor}')

print(obtained_dict)




logs.to_csv('logs.csv')