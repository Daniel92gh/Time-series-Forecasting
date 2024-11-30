import torch 
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.lstm = nn.LSTM(input_size=config.general.feature_size, 
                            hidden_size=config.lstm.hidden_size, 
                            num_layers=config.lstm.num_layers, 
                            bidirectional=False, 
                            batch_first=True)
        self.fc1 = nn.Linear(config.lstm.hidden_size, config.lstm.linear_size)
        self.fc2 = nn.Linear(config.lstm.linear_size, 1)
        self.dropout = nn.Dropout(config.lstm.dropout_ratio)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # lstm layer
        lstm_out , (_, _) = self.lstm(x)
        # last time-step output and drop out layer
        last_out = lstm_out[:, -1, :] # shape : (batch * hidden size) 
        x = self.dropout(last_out)
        # linear layer 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x) 
        x = self.sigmoid(x)
        return x.squeeze(-1)  
