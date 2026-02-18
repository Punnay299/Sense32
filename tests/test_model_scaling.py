
import pytest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.networks import WifiPoseModel, RFEncoder

def test_rf_encoder_scaling():
    """Test if larger RFEncoder initializes and forwards correctly."""
    batch_size = 4
    seq_len = 50
    feats = 2
    
    # Check default big model
    encoder = RFEncoder(input_features=feats, hidden_dim=256, num_layers=3)
    
    x = torch.randn(batch_size, seq_len, feats)
    out = encoder(x)
    
    # Expected output: (batch, hidden_dim)
    assert out.shape == (batch_size, 256)

def test_full_model_scaling():
    """Test full WifiPoseModel with increased capacity."""
    model = WifiPoseModel(input_features=2, output_points=33)
    
    # Check internal dimensions
    # Current code hardcodes hidden_dim=256 in WifiPoseModel init
    # We verify it output correct shapes
    
    
    x = torch.randn(2, 50, 2) # (Batch, Time, Feat)
    pose, pres, loc = model(x)
    
    assert pose.shape == (2, 33*2) # 66
    assert pres.shape == (2, 1)    # 1
    print("All scaling tests passed!")

if __name__ == "__main__":
    test_rf_encoder_scaling()
    test_full_model_scaling()

