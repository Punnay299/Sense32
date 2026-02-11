import sys
import os
import torch
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.dataset import RFDataset
from src.model.networks import WifiPoseModel

logging.basicConfig(level=logging.INFO)

def main():
    print("Starting robustness verification...")
    # Find one session
    base_dir = "data"
    sessions = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if "session_" in d]
    if not sessions:
        print("No sessions found.")
        return

    print(f"Testing with session: {sessions[0]}")
    
    # Init Dataset (Fits scaler)
    ds = RFDataset([sessions[0]], augment=False)
    print(f"Dataset loaded: {len(ds)} samples")
    
    if len(ds) > 0:
        sample = ds[0]
        rf = sample['rf']
        print(f"RF Type: {type(rf)}")
        if isinstance(rf, torch.Tensor):
             print(f"RF Shape: {rf.shape}")
             print(f"RF Min/Max: {rf.min():.2f}/{rf.max():.2f}")
        else:
             print("RF is not a tensor!")
        
    # Test Scaler Save/Load
    ds.scaler.save("models/test_scaler.json")
    print("Scaler saved.")
    
    # Test Inference Model
    model = WifiPoseModel(input_features=64)
    out = model(rf.unsqueeze(0).permute(0, 2, 1)) # (1, 64, 50) -> permute -> (1, 64, 50) wait. 
    # RFDataset output: (64, 50) -> Permute(1, 0) in __getitem__ -> (50, 64) -> Permute back in train loop.
    # RFDataset __getitem__: return rf_tensor.permute(1, 0) -> (Channels, Seq) -> (64, 50).
    # Networks: expects (Batch, Channels, Length).
    # So model(rf.unsqueeze(0)) works directly if RF is (64, 50).
    # Let's check networks.py again.
    # RFEncoder: x = x.permute(0, 2, 1). Input (Batch, Seq, Feat) or (Batch, Feat, Seq)?
    # "CNN expects (Batch, Channels, Length)" -> (Batch, 64, 50).
    # "x = x.permute(0, 2, 1)" -> Input must be (Batch, 50, 64).
    # My train_local.py does: rf = rf.permute(0, 2, 1).
    # My dataset returns: (64, 50).
    # DataLoader stacks -> (Batch, 64, 50).
    # Train loop: rf.permute(0, 2, 1) -> (Batch, 50, 64).
    # Model: x.permute(0, 2, 1) -> (Batch, 64, 50).
    # CNN: (Batch, 64, 50).
    # Correct.
    
    print("Verification Successful.")

if __name__ == "__main__":
    main()
