
import random, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import yfinance as yf, ta
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class LSTM_CNN_Attention(nn.Module):
    def __init__(self, input_size, lstm_hidden=64, cnn_out=24, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.ln   = nn.LayerNorm(2 * lstm_hidden)

        self.conv1 = nn.Conv1d(2*lstm_hidden, cnn_out, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(cnn_out)
        self.conv2 = nn.Conv1d(cnn_out, cnn_out, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(cnn_out)
        self.cnn_drop = nn.Dropout(0.2)

        self.attn_fc  = nn.Linear(cnn_out, 32)
        self.attn_vec = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.3)
        self.head    = nn.Sequential(
            nn.Linear(cnn_out, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        h, _  = self.lstm(x)
        h     = self.ln(h)
        c     = h.permute(0,2,1)
        c     = F.relu(self.bn1(self.conv1(c)))
        c     = F.relu(self.bn2(self.conv2(c)))
        c     = self.cnn_drop(c).permute(0,2,1)

        e  = torch.tanh(self.attn_fc(c))
        w  = F.softmax(self.attn_vec(e), dim=1)
        ctx = torch.sum(w * c, dim=1)
        out = self.dropout(ctx)
        return self.head(out)
