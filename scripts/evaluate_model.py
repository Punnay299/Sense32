import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import logging
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.networks import WifiPoseModel
from src.model.dataset import RFDataset
from src.utils.normalization import AdaptiveScaler

def evaluate():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # 1. Load Data
    base_dir = "data"
    sessions = []
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path) and "session_" in name:
                sessions.append(path)
                
    if not sessions:
        logging.error("No sessions found.")
        return

    logging.info(f"Found {len(sessions)} sessions.")
    
    # 2. Load Scaler
    scaler = AdaptiveScaler()
    if os.path.exists("models/scaler.json"):
        scaler.load("models/scaler.json")
        logging.info("Loaded scaler.")
    else:
        logging.warning("Scaler not found!")
        
    # 3. Create Dataset (No Augmentation)
    dataset = RFDataset(sessions, augment=False, scaler=scaler)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 4. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WifiPoseModel(input_features=320, output_points=33).to(device)
    
    model_path = "models/best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded model from {model_path}")
    else:
        logging.error("Model not found!")
        return

    model.eval()
    
    all_preds = []
    all_gts = []
    
    logging.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch in loader:
            # [Batch, 320, Seq] -> Permute to [Batch, Seq, 320] for Network
            # WAIT. networks.py expects [Batch, Seq, 320] ??
            # Let's check train_local.py
            # train_local.py: rf = rf.permute(0, 2, 1) # [Batch, Seq, 320]
            # Yes.
            
            rf = batch["rf"].to(device).permute(0, 2, 1)
            loc_gt = batch["location"].to(device)
            
            _, _, loc_pred = model(rf)
            
            _, preds = torch.max(loc_pred, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(loc_gt.cpu().numpy())
            
            # Accumulate Features for Stats
            # rf: [Batch, Seq, 320]
            # We want to aggregate by class
            rf_np = rf.cpu().numpy() # [B, S, 320]
            # Average over Seq to get [B, 320]
            rf_mean = np.mean(rf_np, axis=1)
            
            gt_np = loc_gt.cpu().numpy()
            
            if 'class_feats' not in locals():
                class_feats = {0: [], 1: [], 2: [], 3: []}
                
            for i in range(len(gt_np)):
                c = gt_np[i]
                if c in class_feats:
                    class_feats[c].append(rf_mean[i])

    # Per-Class Stats
    target_names = ["Room A", "Room B", "Hallway", "Empty"]
    print("\n--- FEATURE STATS PER CLASS ---")
    
    # Feature Ranges
    # 0-64: Amp A, 128-192: Amp B, 256-320: Diff
    
    for c in sorted(class_feats.keys()):
        if len(class_feats[c]) == 0: continue
        
        feats = np.array(class_feats[c]) # [N, 320]
        
        # Means
        mu_amp_a = np.mean(feats[:, 0:64])
        mu_amp_b = np.mean(feats[:, 128:192])
        mu_diff  = np.mean(feats[:, 256:320])
        
        # Stds
        std_diff = np.std(feats[:, 256:320])
        
        name = target_names[c] if c < len(target_names) else str(c)
        print(f"[{name}] Samples: {len(feats)}")
        print(f"  Amp A Mean: {mu_amp_a:.4f}")
        print(f"  Amp B Mean: {mu_amp_b:.4f}")
        print(f"  Diff Mean:  {mu_diff:.4f} (+/- {std_diff:.4f})")
        print("-" * 30)
            
    # Metrics
    target_names = ["Room A", "Room B", "Hallway", "Empty"]
    unique_labels = sorted(list(set(all_gts)))
    present_names = [target_names[i] for i in unique_labels]
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_gts, all_preds, labels=unique_labels, target_names=present_names, zero_division=0))
    
    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(all_gts, all_preds, labels=unique_labels)
    print(cm)
    
    # Save validation stats to a file for review
    with open("evaluation_results.txt", "w") as f:
        f.write(classification_report(all_gts, all_preds, labels=unique_labels, target_names=present_names, zero_division=0))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print("Results saved to evaluation_results.txt")

if __name__ == "__main__":
    evaluate()
