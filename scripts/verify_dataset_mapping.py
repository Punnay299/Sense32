import sys
import os
import torch
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.dataset import RFDataset

# Setup Logging
logging.basicConfig(level=logging.INFO)

def verify_dataset():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Data directory not found.")
        return

    # Find all session folders
    session_paths = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Filter for just 2 sessions to be fast
    room_a_sessions = [p for p in session_paths if "room_a" in p or "room1" in p]
    room_b_sessions = [p for p in session_paths if "room_b" in p or "room2" in p]
    
    target_sessions = []
    if room_a_sessions: target_sessions.append(room_a_sessions[-1]) # Take latest
    if room_b_sessions: target_sessions.append(room_b_sessions[-1])
    
    if not target_sessions:
        print("Could not find Room A or Room B sessions.")
        return

    print(f"Verifying {len(target_sessions)} sessions: {[os.path.basename(p) for p in target_sessions]}")
    
    # Load Scaler
    scaler_path = "models/scaler.json"
    scaler = None
    if os.path.exists(scaler_path):
        from src.utils.normalization import AdaptiveScaler
        scaler = AdaptiveScaler()
        scaler.load(scaler_path)
        print(f"Loaded Scaler from {scaler_path}")
    else:
        print("WARNING: models/scaler.json not found. Fitting on the fly (might be misleading).")

    # Load Dataset with Scaler
    ds = RFDataset(target_sessions, seq_len=10, scaler=scaler)
    
    print("\n--- Verifying SCALED Variance vs Label ---")
    print("Label Key: 0=Room A, 1=Room B")
    
    node_a_var = []
    node_b_var = []
    labels = []
    
    for i in range(len(ds)):
        sample = ds[i]
        rf = sample['rf'].numpy() # Shape: [256, Seq] -> Actually [256, 10]
        
        # Dataset returns [Channels, Time]
        # Slot A: Channels 0-127 (LogAmp + Phase)
        # Slot B: Channels 128-255 (LogAmp + Phase)
        # Diff: Channels 256-319 (DiffAmp)
        
        slot_a = rf[0:128, :] 
        slot_b = rf[128:256, :]
        diff_feat = rf[256:320, :]
        
        # Calculate Variance (std dev) across the Sequence for this window
        # We want to see how much it 'moves'
        # Or just magnitude? Neural net sees magnitude + shape.
        # Let's look at Mean Absolute Amplitude first, but purely on SCALED data.
        
        # Actually, let's look at the MEAN value of the SCALED data. 
        # Since scaler maps quiet -> 0.5 (if 0-1) or similar? 
        # Scaler maps to 0-1. 
        
        # Let's stick to Energy (Mean Abs) of the Scaled Data.
        # If Scaled Data is correctly separated, Slot A should be 'hotter' in Room A.
        
        amp_a = slot_a[0:64, :]
        amp_b = slot_b[0:64, :]
        
        energy_a = np.mean(np.abs(amp_a))
        energy_b = np.mean(np.abs(amp_b))
        
        label = int(sample['location'].item())
        
        if label in [0, 1]:
            node_a_var.append(energy_a)
            node_b_var.append(energy_b)
            labels.append(label)

    node_a_var = np.array(node_a_var)
    node_b_var = np.array(node_b_var)
    labels = np.array(labels)
    
    print("\n--- Results (SCALED DATA) ---")
    for lbl in [0, 1]:
        mask = labels == lbl
        if np.sum(mask) == 0:
            print(f"No samples for Class {lbl}")
            continue
            
        ea = np.mean(node_a_var[mask])
        eb = np.mean(node_b_var[mask])
        count = np.sum(mask)
        
        print(f"Label {lbl} ({'Room A' if lbl==0 else 'Room B'}): Samples={count}")
        print(f"  Avg Scaled LogEnergy Slot A: {ea:.4f}")
        print(f"  Avg Scaled LogEnergy Slot B: {eb:.4f}")
        
        if lbl == 0:
            if ea > eb: print("  ✅ PASS: Room A is 'hotter' in Slot A.")
            else: print("  ❌ FAIL: Room A is 'hotter' in Slot B.")
        elif lbl == 1:
            if eb > ea: print("  ✅ PASS: Room B is 'hotter' in Slot B.")
            else: print("  ❌ FAIL: Room B is 'hotter' in Slot A.")


if __name__ == "__main__":
    verify_dataset()
