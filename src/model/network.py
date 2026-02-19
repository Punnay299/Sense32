import torch
import torch.nn as nn

class RFModel(nn.Module):
    def __init__(self, input_channels=1, seq_len=100):
        super(RFModel, self).__init__()
        
        # Feature Extractor (Temporal Local patterns)
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Sequence Aggregator
        # Input to RNN: (Batch, Time, Channels)
        # Pool reduces time by 2.
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        
        # Heads
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # x, y
            nn.Sigmoid() # Normalize to 0-1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Presence logit
        )

    def forward(self, x):
        # x: [Batch, Chan, Time]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Prepare for GRU: [Batch, Time, Chan]
        x = x.permute(0, 2, 1)
        
        _, h_n = self.gru(x)
        feat = h_n[-1] # [Batch, Hidden]
        
        coords = self.regressor(feat)
        presence = self.classifier(feat)
        
        return coords, presence
