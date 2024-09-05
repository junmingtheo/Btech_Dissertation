import torch
import torch.nn as nn

class GRUNet(nn.Module):
    
    def __init__(self,input_len,hidden_size_1,hidden_size_2,out_len):
        
        super().__init__()
        
        self.hidden_size_1 = hidden_size_1
        
        self.hidden_size_2 = hidden_size_2
        
        self.input_size = input_len
        
        self.GRU_1 = nn.GRU(input_len,hidden_size_1,batch_first=True)
        
        self.GRU_2 = nn.GRU(hidden_size_1,hidden_size_2, batch_first=True)
        
        self.out_layer = nn.Linear(hidden_size_2,out_len)
        
        
        
    def forward(self,X):
        hidden_state_1 = torch.zeros(1,X.size(0),self.hidden_size_1)
        hidden_state_2 = torch.zeros(1,X.size(0),self.hidden_size_2)


        out_1 , _ = self.GRU_1(X,hidden_state_1)
        
        out_2 , _ = self.GRU_2(out_1,hidden_state_2)
        
        pred = self.linear(out_2.view(len(self.)))
        
        return pred
        
class modelgen():

    def __init__(self):
        pass

