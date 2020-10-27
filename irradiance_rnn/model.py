import torch
import torch.nn as nn


class LSTMDrop(nn.Module):

    def __init__(self, hidden_size, num_layers, dropout, device):
        super(LSTMDrop, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc1(out[:, -1, :])
        return out
