
import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding

class matrix_factorization(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super(matrix_factorization, self).__init__()

        self.P = nn.Embedding(num_embeddings = num_users, embedding_dim = num_factors)
        self.Q = nn.Embedding(num_embeddings = num_items, embedding_dim = num_factors)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.init_weights()
    def forward(self, user_id, item_id):
        p_u = self.P(user_id)
        q_i = self.Q(item_id)
        #print(p_u.shape)
        #print(q_i.shape)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        #print(f'{b_u.shape} {b_i.shape}')
        relevance = torch.sum(p_u * q_i, axis = 1, keepdim=True)
        #print(f'relevance {relevance.shape}')
        outputs =  relevance + b_u + b_i
        #print(outputs)
        #print(outputs.shape)
        return outputs
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


#print(help(nn.Embedding))

net = matrix_factorization(20,5,10)
