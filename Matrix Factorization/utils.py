import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


#help(nn.MSELoss())
def plot(train_loss, test_loss):
    x = [*range(1, len(train_loss))]
    plt.plot(x, train_loss[1:], label = 'train_loss')
    plt.plot(x, test_loss[1:], label = 'test_loss')
    plt.legend()
    plt.grid()
    plt.savefig('train_test_loss.png')
    plt.clf()