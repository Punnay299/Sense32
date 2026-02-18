import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze():
    # Load Data
    try:
        rec = np.loadtxt("debug_recorded_features.csv", delimiter=",")
        live = np.loadtxt("debug_live_features.csv", delimiter=",")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Handle shapes
    # Recorded: [Seq, 320] (e.g., 60, 320) or [320] if 1D? 
    # Logic in evaluate_model.py: sample_feat = rf[0].cpu().numpy() -> [Seq, 320]
    
    # Live: [50, 320]
    
    print(f"Recorded Shape: {rec.shape}")
    print(f"Live Shape:     {live.shape}")
    
    # Feature Blocks
    # 0-64:    Amp A (Log)
    # 64-128:  Phase A
    # 128-192: Amp B (Log)
    # 192-256: Phase B
    # 256-320: Diff (Amp A - Amp B)
    
    blocks = {
        "Amp A (Log)": (0, 64),
        "Phase A":     (64, 128),
        "Amp B (Log)": (128, 192),
        "Phase B":     (192, 256),
        "Diff Feat":   (256, 320)
    }
    
    print("\n--- FEATURE COMPARISON (MEAN VALUES) ---")
    print(f"{'Block':<15} | {'Recorded Mean':<15} | {'Live Mean':<15} | {'Diff':<10}")
    print("-" * 65)
    
    for name, (start, end) in blocks.items():
        # Flatten to 1D for global stats
        r_block = rec[:, start:end].flatten()
        l_block = live[:, start:end].flatten()
        
        r_mean = np.mean(r_block)
        l_mean = np.mean(l_block)
        diff = l_mean - r_mean
        
        print(f"{name:<15} | {r_mean:<15.4f} | {l_mean:<15.4f} | {diff:<10.4f}")

    print("\n--- DETAILED STATS (Recorded vs Live) ---")
    for name, (start, end) in blocks.items():
        r_block = rec[:, start:end]
        l_block = live[:, start:end]
        
        print(f"\n[{name}]")
        print(f"  Rec: Min={r_block.min():.4f}, Max={r_block.max():.4f}, Std={r_block.std():.4f}")
        print(f"  Live: Min={l_block.min():.4f}, Max={l_block.max():.4f}, Std={l_block.std():.4f}")

if __name__ == "__main__":
    analyze()
