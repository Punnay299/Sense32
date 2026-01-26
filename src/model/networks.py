try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class MockNN:
        class Module:
            def __init__(self): pass
            def __call__(self, *args, **kwargs): pass
    nn = MockNN() # Dummy namespace

class RFEncoder(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_features=2, hidden_dim=64, num_layers=2):
        """
        :param input_features: Number of RF channels (e.g. RSSI, RTT)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.")
        super(RFEncoder, self).__init__()
        
        # Feature Extraction (Spatial/Low-level)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Temporal Modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
    def forward(self, x):
        """
        :param x: (Batch, Seq_Len, Features)
        """
        # CNN expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        
        # LSTM expects (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        
        # Return last hidden state
        # hn shape: (num_layers, batch, hidden_dim)
        return hn[-1]

class PoseRegressor(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim=64, output_points=17):
        if not TORCH_AVAILABLE: raise ImportError("PyTorch missing")
        super(PoseRegressor, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_points * 2) # X, Y for each point
        )
        
    def forward(self, x):
        """
        :param x: Latent vector from Encoder
        :return: (Batch, Points*2) flattened
        """
        return self.fc(x)

class PresenceDetector(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim=64):
        super(PresenceDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fc(x)

class WifiPoseModel(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_features=2, output_points=17):
        if not TORCH_AVAILABLE: raise ImportError("PyTorch missing")
        super(WifiPoseModel, self).__init__()
        self.encoder = RFEncoder(input_features=input_features)
        self.pose_head = PoseRegressor(input_dim=64, output_points=output_points)
        self.presence_head = PresenceDetector(input_dim=64)
        
    def forward(self, x):
        latent = self.encoder(x)
        pose = self.pose_head(latent)
        presence = self.presence_head(latent)
        return pose, presence
