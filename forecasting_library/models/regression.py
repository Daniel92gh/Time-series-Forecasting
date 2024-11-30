import torch 
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.linear = nn.Linear(config.general.feature_size, 1)

    def forward(self, x):
        # linear layer
        x = self.linear(x) 
        return x.squeeze(-1)    
    