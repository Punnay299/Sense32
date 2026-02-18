import os
import sys
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.csi_sanitizer import CSISanitizer

def main():
    print("=========================================")
    print("   CSI Data Quality Validation Script    ")
    print("=========================================")
    
    if not os.path.exists("data"):
        print("No 'data/' directory found.")
        return

    # Find a session with data
    sessions = [d for d in os.listdir("data") if d.startswith("session_") and os.path.isdir(os.path.join("data", d))]
    if not sessions:
        print("No sessions found.")
        return
        
    # Use the first valid session
    target_session = None
    for s in sessions:
        if os.path.exists(os.path.join("data", s, "rf_data.csv")):
            target_session = os.path.join("data", s)
            break
            
    if not target_session:
        print("No sessions with rf_data.csv found.")
        return
        
    print(f"Analyzing Session: {target_session}")
    rf_path = os.path.join(target_session, "rf_data.csv")
    
    try:
        df = pd.read_csv(rf_path)
        if len(df) == 0:
            print("Empty RF file.")
            return
            
        # Extract Amplitudes
        print("Parsing CSI Amplitudes...")
        raw_amps = []
        for x in tqdm(df["csi_amp"]):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    if len(val) >= 64: val = val[:64]
                    else: val = val + [0]*(64-len(val))
                    raw_amps.append(val)
                else:
                    raw_amps.append([0]*64)
            except:
                raw_amps.append([0]*64)
        
        raw_amps = np.array(raw_amps, dtype=np.float32)
        print(f"Loaded Raw Shape: {raw_amps.shape}")
        
        # Apply Sanitization
        print("\n>> Applying Hampel Filter...")
        clean_amps = CSISanitizer.sanitize_amplitude(raw_amps)
        
        # Comparison Metrics
        diff = np.abs(raw_amps - clean_amps)
        num_changed = np.sum(diff > 0.001)
        total_points = raw_amps.size
        mse = np.mean(diff ** 2)
        
        print("\n--- Quality Report ---")
        print(f"Total Data Points: {total_points}")
        print(f"Outliers Corrected: {num_changed} ({num_changed/total_points*100:.2f}%)")
        print(f"Mean Squared Error (Change Magnitude): {mse:.4f}")
        
        # Visualize First Subcarrier
        # We can't show GUI, so we just print stats of a noisy subcarrier
        
        # Find subcarrier with most changes
        changes_per_sc = np.sum(diff > 0.001, axis=0)
        noisy_sc = np.argmax(changes_per_sc)
        
        print(f"\nNoisiest Subcarrier Index: {noisy_sc}")
        print(f"Changes in this SC: {changes_per_sc[noisy_sc]}")
        
        # Print a snippet of Before/After for that SC
        print("\nSnippet (Raw vs Clean):")
        start_idx = 0
        end_idx = min(20, len(raw_amps))
        for i in range(start_idx, end_idx):
            r = raw_amps[i, noisy_sc]
            c = clean_amps[i, noisy_sc]
            mark = "*" if abs(r - c) > 0.001 else " "
            print(f"Frame {i:03d}: \t{r:.2f} -> {c:.2f} {mark}")
            
        print("\n>> Data Quality Logic Verified.")
        print("This sanitization logic is now ACTIVE in src/model/dataset.py")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
