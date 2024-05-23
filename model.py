import torch.nn as nn
import torch

class CharRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, drop_prob):
        super(CharRNN,self).__init__()
        
        self.input_dim = input_dim # dic_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob= drop_prob
        
        self.rnn = torch.nn.RNN(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.drop_prob, batch_first=True)
        self.fc=nn.Linear(self.hidden_dim, self.input_dim)
        
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, drop_prob):
        super(CharLSTM,self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.drop_prob, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input,hidden)
        output= self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):       
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim), torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return initial_hidden
    