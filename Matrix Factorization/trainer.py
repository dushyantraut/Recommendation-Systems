import torch
import torch.nn as nn
import data
from torch.utils.data import DataLoader
import model
import utils
import tqdm
######################       HYPERPARAMETERS       #############


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class trainer():
    def __init__(self, train_dataset, test_dataset, batch_size, lr, n_epoch, n_factor, n_user, n_item, weight_decay):
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lr = lr
        self.n_factor = n_factor
        self.n_user = n_user
        self.n_item = n_item
        self.weight_decay = weight_decay


        self.train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
        self.test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size)

        self.net = model.matrix_factorization(num_factors = self.n_factor, num_users = self.n_user, num_items= self.n_item).to(device)
        self.criterion = utils.RMSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        self.train_rmse = []
        self.test_rmse = []
        
    # to cheeck correctness of data
    #print(test[:5])





    '''
    for name, params in net.named_parameters():
        print(name, params.shape)
    '''

    def train_one_epoch(self):
        losses = []
        for user, item , rating in tqdm.tqdm(self.train_loader):
            #print(x.shape,y.shape,z.shape)
            user,item, rating = user.to(device), item.to(device), rating.to(device)
            
            out = self.net(user,item).reshape(rating.shape) #shape -- batch_size * 1
            loss = self.criterion(out, rating)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss)
        return losses

    


    def check_test_loss(self):
        losses = []
        for user,item,rating in self.test_loader:
            user,item, rating = user.to(device), item.to(device), rating.to(device)
            out = self.net(user,item).reshape(rating.shape) #shape -- batch_size * 1
            loss = self.criterion(out, rating)
            losses.append(loss)
        return losses


    def train(self):
        best_params = {
            'min_loss' : 10,
            'epoch' : 0,
            'lr' : 0,
            'n_factor':0,
            'batch_size':0

        }
        print('Training Started')
        
        for epoch in range(self.n_epoch):
            losses = self.train_one_epoch()
            with torch.no_grad():
                print(f'train loss after {epoch} epoch is {sum(losses)/len(losses)}')
                self.train_rmse.append( ( sum(losses)/len(losses) ).item())
                #print(len(losses))
                test_loss = self.check_test_loss()
                print(f'test loss after {epoch} epoch is {sum(test_loss)/len(test_loss)}')
                self.test_rmse.append( ( sum(test_loss)/len(test_loss) ).item() )
                #print(len(test_loss))
                if(best_params['min_loss'] > self.test_rmse[-1]):
                    
                    best_params['min_loss'] =  self.test_rmse[-1]
                    best_params['epoch'] = epoch
                    best_params['batch_size'] = self.batch_size
                    best_params['n_factor'] = self.n_factor
                    best_params['lr'] = self.lr
        
        print(f"best test loss was found to be {best_params['min_loss']}")
        print(best_params)
        return best_params, self.train_rmse, self.test_rmse
    




######################you can comment this part for hypertuning###############



device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 100
n_epoch = 20
lr = 0.001
dropout = 0.5
n_factor = 30
weight_decay = 0.0001


train, test, n_users, n_items = data.train_test_split(0.1)
print(f'training set contains {train.shape[0]}')
print(f'test set contains {test.shape[0]}')
train_dataset = data.my_dataset(train)
test_dataset = data.my_dataset(test)



our_model_trainer = trainer(train_dataset = train_dataset,
                                        test_dataset = test_dataset,
                                        batch_size = batch_size,
                                        lr = lr,
                                        n_epoch = n_epoch,
                                        n_factor = n_factor,
                                        n_user = n_users,
                                        n_item = n_items,
                                        weight_decay = weight_decay)

best_params, train_losses, test_losses = our_model_trainer.train()



utils.plot(train_losses, test_losses)






